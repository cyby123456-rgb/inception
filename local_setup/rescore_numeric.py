"""Post-hoc numeric rescoring; preserve raw benchmark scores and extraction source."""
import re
from fractions import Fraction

NUMBER=r'-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?'


def extract_numeric(text):
    # Horizontal whitespace only. An empty 'Final Answer:' heading is not an answer.
    patterns=[('answer_line',r'answer[^\S\n]*:[^\S\n]*([^\n]*)'),
              ('hash_line',r'####[^\S\n]*([^\n]*)'),('boxed',r'\\boxed\{([^{}]+)\}')]
    for label,pattern in patterns:
        for candidate in reversed(re.findall(pattern,text,flags=re.I)):
            nums=re.findall(NUMBER,candidate.replace(',','').replace('−','-'))
            if nums:return nums[-1],label
    nums=re.findall(NUMBER,text.replace(',','').replace('−','-'))
    return (nums[-1],'last_number_fallback') if nums else (None,'unparsed')


def equivalent(a,b):
    try:return Fraction(a)==Fraction(b)
    except (ValueError,TypeError,ZeroDivisionError):return False


def score(text,answer):
    value,method=extract_numeric(text)
    return dict(extracted_numeric=value,extraction_method=method,numeric_correct=equivalent(value,answer))
