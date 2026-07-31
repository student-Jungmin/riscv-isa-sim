// vfclass.v vd, vs2, vm
VI_VFP_V_LOOP
({
  vd.v = f16_classify(vs2);
},
{
  vd.v = f32_classify(vs2);
},
{
  vd.v = f64_classify(vs2);
})
// Deliberately no e8 arm: the class mask is 10 bits and an SEW=8 destination is 8,
// so the NaN bits 8 and 9 have nowhere to go and every NaN would silently classify
// as nothing. RVV assumes SEW >= 16 here. Traps on the macro's default instead.
