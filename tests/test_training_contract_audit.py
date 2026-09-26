import copy
import json
from pathlib import Path
import sys

import pytest
import yaml
from llamafactory.hparams import FinetuningArguments

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'local_setup'),str(ROOT/'scripts')]
from training_contract import validate_scope, make_contract, validate_resume, validate_initialization_layout, sha
from qwen3_4b_recipe import architecture, stage_config
from benchmark_trained_target import command
from checkpoint_identity import target_adapter_signature
from matrix_audit import write_protocol_manifest, audit_suite
from joint_training import joint_config
from prepare_joint_from_base import validate_model
from test_qwen3_4b_matrix import recipe, config


def cfg(scope='target_t_head'):
    r=recipe()
    stage={'target_t':'core','head':'head','t_head':'joint','target_t_head':'joint'}[scope]
    return stage_config('model','data','out',42,stage,4000,architecture(config(),r),r,scope in ('target_t','target_t_head'))


@pytest.mark.parametrize('scope',['target_t','head','t_head','target_t_head'])
def test_explicit_scopes_and_freeze_flags(scope):
    c=cfg(scope)
    d=validate_scope(c,scope)
    assert d['original_base_trainable'] is False
    assert d['target_lora_trainable']==(scope in ('target_t','target_t_head'))
    bad=dict(c,recurft_stage1_heads_only=not c['recurft_stage1_heads_only'])
    with pytest.raises(ValueError,match='Freeze flags'):
        validate_scope(bad,scope)


@pytest.mark.parametrize('change',[
    {'recurft_recurrent_loss_weight':0},
    {'recurft_multistep_loss_weight':0},
    {'recurft_multistep_steps':1},
    {'recurft_multistep_boundary_logit_kl_loss_weight':0},
    {'recurft_t_lora_rank':0},
    {'recurft_multistep_residual_rank':2},
])
def test_joint_cannot_silently_disable_required_training_paths(change):
    c=cfg();c.update(change)
    with pytest.raises(ValueError):
        validate_scope(c,'target_t_head')
    fields=FinetuningArguments.__dataclass_fields__
    with pytest.raises(ValueError):
        FinetuningArguments(**{k:v for k,v in c.items() if k in fields})


def test_exact_resume_rejects_scope_objective_data_and_budget_changes(tmp_path):
    c=cfg()
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}')
    data=tmp_path/'data';data.mkdir()
    (data/'train.json').write_text('[]')
    (data/'dataset_info.json').write_text(json.dumps({'matrix_train':{'file_name':'train.json'}}))
    c.update(model_name_or_path=str(model),dataset_dir=str(data))
    original=make_contract(c,'target_t_head')
    (tmp_path/'training_contract.json').write_text(json.dumps(original))
    assert validate_resume(tmp_path,make_contract(dict(c,output_dir='new',logging_steps=1),'target_t_head'),True)=='exact_contract_match'
    for change in [{'seed':43},{'max_steps':5000},{'learning_rate':1e-4},{'recurft_boundary_teacher_source':'reference'}]:
        with pytest.raises(ValueError,match='training contract'):
            validate_resume(tmp_path,make_contract(dict(c,**change),'target_t_head'),True)
    (data/'train.json').write_text('[1]')
    with pytest.raises(ValueError,match='training contract'):
        validate_resume(tmp_path,make_contract(c,'target_t_head'),True)
    (tmp_path/'training_contract.json').unlink()
    with pytest.raises(ValueError,match='requires training_contract'):
        validate_resume(tmp_path,original,True)
    assert validate_resume(tmp_path,original,False)=='legacy_checkpoint_without_contract'


def test_transition_checks_layout_scaling_and_only_allows_head_addition(tmp_path):
    c=cfg('head')
    meta={key:c['recurft_'+key] for key in ['loop_start_layer','loop_end_layer','pre_lora_rank','last_lora_rank','t_lora_rank']}
    meta.update(boundary_head_rank=0,t_lora_alpha=c['recurft_t_lora_alpha'])
    (tmp_path/'recurft_config.json').write_text(json.dumps(meta))
    validate_initialization_layout(tmp_path,c,'head')
    with pytest.raises(ValueError,match='add a new boundary'):
        validate_initialization_layout(tmp_path,c,'t_head')
    with pytest.raises(ValueError,match='layout mismatch'):
        validate_initialization_layout(tmp_path,dict(c,recurft_loop_start_layer=32),'head')
    with pytest.raises(ValueError,match='scaling mismatch'):
        validate_initialization_layout(tmp_path,dict(c,recurft_t_lora_alpha=80),'head')


@pytest.mark.parametrize('override',['--before','--after','--bef','--aft','--before=other','--aft=other'])
def test_single_target_rejects_overrides_and_abbreviations(tmp_path,override):
    for name in ['adapter_model.safetensors','recurft_recurrent.safetensors','recurft_config.json']:
        (tmp_path/name).touch()
    with pytest.raises(SystemExit):
        command(['--checkpoint',str(tmp_path),override,'other'])


def test_target_identity_includes_adapter_scaling(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b';a.mkdir();b.mkdir()
    base=dict(r=8,lora_alpha=16,target_modules=['q_proj'],base_model_name_or_path='/old')
    (a/'adapter_config.json').write_text(json.dumps(base))
    (b/'adapter_config.json').write_text(json.dumps(dict(base,base_model_name_or_path='/new')))
    assert target_adapter_signature(a)==target_adapter_signature(b)
    (b/'adapter_config.json').write_text(json.dumps(dict(base,lora_alpha=32)))
    assert target_adapter_signature(a)!=target_adapter_signature(b)


def test_protocol_audit_detects_config_and_data_mutation(tmp_path):
    c=cfg()
    d=tmp_path/'full_joint/seed_42/joint';d.mkdir(parents=True)
    c['output_dir']=str(d/'checkpoints')
    (d/'train.yaml').write_text(yaml.safe_dump(c))
    stage=dict(scope='target_t_head',name='joint',config=str(d/'train.yaml'),directory=str(d),steps=4000,training_regime=validate_scope(c,'target_t_head'))
    data=tmp_path/'data';data.mkdir()
    for name in ['dataset_info.json','train.json','validation.json','dev.jsonl','test.jsonl']:(data/name).write_text('{}')
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}')
    source=tmp_path/'source_snapshot';source.mkdir();(source/'code.py').write_text('original')
    (tmp_path/'source_manifest.json').write_text(json.dumps({'code.py':sha(source/'code.py')}))
    plan=dict(model=str(model),validation_inference=str(data/'dev.jsonl'),test_data=str(data/'test.jsonl'),
              experiments=[dict(arm='full_joint',seed=42,stages=[stage])])
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    write_protocol_manifest(tmp_path,plan)
    assert audit_suite(tmp_path,plan)['protocol_verified']
    original=(d/'train.yaml').read_text()
    (d/'train.yaml').write_text(original+'# changed\n')
    with pytest.raises(ValueError,match='protocol/data/config modified'):
        audit_suite(tmp_path,plan)
    (d/'train.yaml').write_text(original)
    (data/'train.json').write_text('[]')
    with pytest.raises(ValueError,match='protocol/data/config modified'):
        audit_suite(tmp_path,plan)


def test_core_and_multistep_retain_original_loss_details():
    r=recipe();arch=architecture(config(),r)
    core=stage_config('model','data','out',42,'core',10,arch,r,True)
    multi=stage_config('model','data','out',42,'multistep',10,arch,r,True)
    assert core['recurft_recurrent_warmup_steps']==200
    assert multi['recurft_recurrent_warmup_steps']==0
    assert multi['recurft_multistep_delta_cosine_loss_weight']==.02


def test_legacy_8b_recipe_cannot_be_silently_used_for_4b():
    with pytest.raises(ValueError,match='8B-specific'):
        validate_model(config())
    validate_model(dict(config(),hidden_size=4096,intermediate_size=12288,tie_word_embeddings=False))
