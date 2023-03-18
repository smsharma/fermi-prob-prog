;+
; NAME:
;   fermi_psc
;
; PURPOSE:
;   Write FERMI point source mask
;
; CALLING SEQUENCE:
;   fermi_psc, nside
;
; INPUTS:
;   nside   - HealPix Nside  (default 256)
;   fwhm    - fwhm [arcmin]  (default 120)
;
; OUTPUTS:
;   <file> $FERMI_DATA/foregrounds/pscmask-fwhm120-256.fits
;
; OPTIONAL OUTPUTS:
;   
; COMMENTS:
;   based on wmap_psc.pro
;   The mask is large enough to cover 95% of the flux of a Gaussian 
;     PSF with the given FWHM.  That is, the FWHM keyword does NOT
;     give the size of the mask circles, but rather gives the FWHM
;     PSF for which this mask is appropriate.
;
;   sqrt(2*alog(1./(1-0.95))) = 2.44775
;   so 2.44775 sigma gives 95% flux containment for a Gaussian.
;
; REVISION HISTORY:
;   2009-Sep-07   Written by Douglas Finkbeiner, CfA
;   2014-Sep-15   Modified by Nick Rodd, MIT, to help outputting multiple masks
;   2015-June-19  Modified by Tracy Slatyer, MIT, to set the mask
;   radius directly, and mask all sources not just bright ones
;   2015-July-2   Updated to Pass8 by Nick Rodd, MIT
;   2019-Nov-15 Updated to 4FGL by Nick Rodd, MIT
;----------------------------------------------------------------------
pro fermi_psc_all_pass8_nick_4fgl, clrad=clrad, clname=clname, nside=nside, energy=energy, classname=classname

  ;if n_params() EQ 0 then begin
  ;   print, 'fermi_psc, nside, fwhm=fwhm, brightest=brightest'
  ;   return
  ;endif 

  if NOT keyword_set(nside) then nside = 256L

; -------- use 3FGL
  fname = '../../data/psc/gll_psc_v31.fit'

  psc = mrdfits(fname, 1)

; -------- healpix coords
  healgen_lb, nside, l, b
  uv = ll2uv([[l], [b]])
  
  p = bytarr(n_elements(b))

; -------- Big object mask    LMC,  SMC,  Orion,  NGC 5090
;            last two are: 3C 454.3, LAT PSR J1836+5925, and Geminga

  lbig = [279.00, 302.30, 207, 309.15, 86.12,  88.8, 195.13]
  bbig = [-33.60, -44.7, -17, 18.91, -38.18, 25.0, 4.27]
  rad = [5, 2, 6, 4, 1.5, 1.5, 2]

; -------- Read objects chosen by eye
  readcol, '../../data/psc/ptsource.dat', leye, beye
  reye = dblarr(n_elements(beye))

; -------- Append big objects to Fermi PSC
  pscl = [psc.glon, leye, lbig]
  pscb = [psc.glat, beye, bbig]
  pscr = [dblarr(n_elements(psc)), reye, rad] ; [deg]

  uvpsc = ll2uv([[pscl], [pscb]])

; -------- Define mask radius
  maskrad = clrad/60d ; 95% containment is 2.448 sigma - use scale to manually adjust the size of the mask
    
  for i=0L, n_elements(pscb)-1 do begin 
     cs = uv#transpose(uvpsc[i, *])
     w = where(cs GT cos(sqrt(maskrad^2+pscr[i]^2)/!radeg))
     p[w] = 1B
     if (i mod 10) eq 0 then print, i
  endfor 
  
  npix = 12L * nside*nside

  mkhdr, phdr, p
  sxaddpar, phdr, 'RADIUS',      maskrad, ' mask circle radius [deg]'
  get_date, date, /timetag
  sxaddpar, phdr, 'DATE', date, ' Creation date'
  sxaddpar, phdr, 'PIXTYPE',  'HEALPIX ', ' Pixel algorithm'
  sxaddpar, phdr, 'ORDERING',   'RING  ', ' Ordering scheme'
  sxaddpar, phdr, 'NSIDE',         nside, ' Resolution parameter'
  sxaddpar, phdr, 'NPIX',           npix, ' # of pixels'
  sxaddpar, phdr, 'FIRSTPIX',          0, ' First pixel (0 based)'
  sxaddpar, phdr, 'LASTPIX',      npix-1, ' Last pixel (0 based)'
  sxaddpar, phdr, 'HISTORY', ' Written by Tracy Slatyer, MIT'
 
  ;file_mkdir, '$FERMI_DATA/mask'
  ; Temporarily write here as I don't have permission to write in Tracy's directory
  prefix = 'Allpscmask_4FGL'
  outname = string('../../data/psc/'+prefix+'-energy' + strtrim(string(energy),2) +clname + classname + '.fits');,$
                   ;format='(A,I3.3,A,I3.3,A)')
  mwrfits, p, outname, phdr, /create
  
  print, outname, ' written.'

  return
end
