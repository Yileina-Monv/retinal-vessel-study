"""Proposed prefix contract: exact inputs/recipe, numeric differences diagnostic."""
import math

class PrefixViolation(RuntimeError):pass

def validate_prefix(actual,expected):
    if not 1<=len(actual)<=30 or len(expected)<len(actual):raise PrefixViolation('invalid prefix length')
    diagnostics=[]
    for number,(a,b) in enumerate(zip(actual,expected),1):
        if a['attempt']!=number or b['attempt']!=number:raise PrefixViolation('nonsequential event')
        for key in ['ids','input_sha256','target_sha256','lr']:
            if a[key]!=b[key]:raise PrefixViolation('input/recipe drift:'+key)
        loss=a['loss'];norm=a['unscaled_preclip_L2'];before=a['scale_before'];after=a['scale_after']
        if not isinstance(loss,(int,float)) or not math.isfinite(loss) or not all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in [before,after]):raise PrefixViolation('nonfinite Loss/AMP scale')
        if a['classification']=='SUCCESS':
            if not isinstance(norm,(int,float)) or not math.isfinite(norm) or a['optimizer_calls']!=1 or a['model_changed'] is not True or after<before:raise PrefixViolation('invalid successful update')
        elif a['classification']=='AMP_SKIP':
            if a['optimizer_calls']!=0 or a['model_changed'] is not False or a['optimizer_changed'] is not False or not after<before:raise PrefixViolation('unhandled or corrupt AMP skip')
        else:raise PrefixViolation('fatal/unknown event; retain ordinary stop rules')
        row={'attempt':number,'same_inputs_targets_ids_LR':True,'actual_classification':a['classification'],'reference_classification':b['classification'],'Loss_abs_difference':abs(loss-b['loss']),'AMP_scales_exact_to_reference':before==b['scale_before'] and after==b['scale_after'],'optimizer_calls_exact_to_reference':a['optimizer_calls']==b['optimizer_calls']}
        if isinstance(norm,(int,float)) and math.isfinite(norm) and isinstance(b['unscaled_preclip_L2'],(int,float)) and math.isfinite(b['unscaled_preclip_L2']):row['gradient_L2_abs_difference']=abs(norm-b['unscaled_preclip_L2'])
        else:row['gradient_L2_abs_difference']=None
        diagnostics.append(row)
    return {'status':'PREFIX_INPUT_RECIPE_AND_INTRINSIC_SAFETY_PASS','numeric_bitwise_equality_required':False,'new_tolerance_scan':False,'events':diagnostics}
