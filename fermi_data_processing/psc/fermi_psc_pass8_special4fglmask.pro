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
;   2015-Jun-26   Adapted to Pass 8 by Tracy Slatyer, MIT
;----------------------------------------------------------------------
pro fermi_psc_pass8_special4fglmask, clrad=clrad, nside=nside, maskname=maskname, maskmap=maskmap

  if NOT keyword_set(nside) then nside = 512L

; -------- load catalog
  

  psc = mrdfits(fname, 1)

  tslist= total(psc.sqrt_ts_band^2,1)

  faint=where(tslist LT 50)
  bright=where(tslist GE 50)

; -------- healpix coords
  healgen_lb, nside, l, b
  uv = ll2uv([[l], [b]])
  maskmap = bytarr(n_elements(b))

  pscl = psc.glon
  pscb = psc.glat

  uvpsc = ll2uv([[pscl], [pscb]])

; -------- Define mask radius
  dimrad = clrad/60.0 ; clrad should describe the desired containment radius (e.g. 95% containment) - use scale to manually adjust the size of the mask

  if keyword_set(scale) then dimrad *=scale

  brightrad=dimrad*3d

  maskrad=dblarr(n_elements(psc))
  maskrad[faint]=dimrad
  maskrad[bright]=brightrad

  for i=0L, n_elements(psc)-1 do begin
     cs = uv#transpose(uvpsc[i, *])
     w = where(cs GT cos(maskrad[i]/!radeg))
     maskmap[w] = 1B
     if (i mod 10) eq 0 then print, i
  endfor 

  npix = 12L * nside*nside

  mkhdr, phdr, maskmap
  sxaddpar, phdr, 'RADIUS',     dimrad, ' mask circle radius [deg]'
  get_date, date, /timetag
  sxaddpar, phdr, 'DATE', date, ' Creation date'
  sxaddpar, phdr, 'PIXTYPE',  'HEALPIX ', ' Pixel algorithm'
  sxaddpar, phdr, 'ORDERING',   'RING  ', ' Ordering scheme'
  sxaddpar, phdr, 'NSIDE',         nside, ' Resolution parameter'
  sxaddpar, phdr, 'NPIX',           npix, ' # of pixels'
  sxaddpar, phdr, 'FIRSTPIX',          0, ' First pixel (0 based)'
  sxaddpar, phdr, 'LASTPIX',      npix-1, ' Last pixel (0 based)'
  sxaddpar, phdr, 'HISTORY', ' Written by D. Finkbeiner, Harvard CfA'

  outname = maskname
  if keyword_set(scale) then outname += '_scale='+strtrim(string(scale),2)
  outname += '.fits'
  ; mwrfits, maskmap, outname, phdr, /create
  
  print, outname, ' written.'

  return
end
