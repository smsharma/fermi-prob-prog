pro make_4fgl_masks_pass8

earr = 2.

clval = fermi_cl_psf_pass8('ULTRACLEANVETO',earr,cl=0.95,eventtype=3)*60d ; Multiply by 60 as the code takes a value in arcminutes

fermi_psc_all_pass8_nick_4fgl,clrad=clval,energy=earr,clname='_0.95_',classname='ULTRACLEANVETO_bestpsf',nside=512L

return
end
