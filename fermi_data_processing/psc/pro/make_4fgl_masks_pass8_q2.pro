pro make_4fgl_masks_pass8_q2
; 2nd quartile of data is eventtype=4 (bestpsf is 3)
; Note Tracy's convention: eventtype 1 and 2 are front and back, 3-6 are best, 2nd, 3rd, 4th

earr = 2.

clval = fermi_cl_psf_pass8('ULTRACLEANVETO',earr,cl=0.95,eventtype=4)*60d ; Multiply by 60 as the code takes a value in arcminutes

fermi_psc_all_pass8_nick_4fgl,clrad=clval,energy=earr,clname='_0.95_',classname='ULTRACLEANVETO_q2',nside=512L

return
end
