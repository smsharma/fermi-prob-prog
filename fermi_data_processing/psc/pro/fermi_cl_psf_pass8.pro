function fermi_cl_psf_pass8,psfname,energy,cl=cl,eventtype=eventtype,fwhm=fwhm,irfs=irfs

  if not keyword_set(cl) then cl = 0.95
  if keyword_set(fwhm) then cl = 0.68 ;get FWHM for Gaussian matching 68% containment radius 

  x = (findgen(2500)+0.5)/100

  psf=fermi_psf_pass8(psfname, x, energy*1000d, eventtype=eventtype,irfs=irfs)

  sum = total(psf*x, /cum)/total(psf*x)
  clval = interpol(x, sum, cl)

  if keyword_set(fwhm) then clval *= sqrt(4*alog(4))/sqrt(2*alog(1./(1-0.68)))

  return,clval
end
