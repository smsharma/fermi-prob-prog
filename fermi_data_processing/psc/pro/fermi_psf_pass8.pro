;+
; NAME:
;   fermi_psf
;
; PURPOSE:
;   Evaluate simple Fermi PSF model given in the Cicerone
;
; CALLING SEQUENCE:
;   psf = fermi_psf(dangle, energy, front=front, back=back)
;
; INPUTS:
;   dangle  - angle from true center [deg]
;   energy  - energy [MeV]
;
; KEYWORDS:
;   front   - front converting
;   back    - back converting
;   
; OUTPUTS:
;   The psf (normalized how?)
; 
; OPTIONAL OUTPUTS:
;   
; EXAMPLES:
;   
; COMMENTS:
;   This routine uses the simple parameterization in the Cicerone. 
;   To get the best (position-dependent) estimate of the PSF, using the
;     full livetime cube, please use fermi_gtpsf.pro
; 
;   Evaluate fermi psf from x, sigma, gamma_core, gamma_tail from files in
;  
;   $FERMI_DIR/refdata/fermi/caldb/CALDB/data/glast/lat/bcf/psf
;
; For an explanation, see the Cicerone:
;   http://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_LAT_IRFs/IRF_PSF.html
;
; REVISION HISTORY:
;   01/10/10 - Written by Douglas Finkbeiner, CfA
;
;----------------------------------------------------------------------
function fermi_psf_base, x, sigma, gamma

  base = (1.d - 1.d/gamma) * (1.d + 0.5d*x^2/(gamma*sigma^2))^(-gamma)/(2d*!dpi*sigma^2)

  w = where(finite(base) eq 0, nbad)
  if nbad NE 0 then message, 'called with bad parameters'

  return, base
end



function fermi_psf_with_tail, x, score, gcore, Ntail, stail, gtail
  fcore = 1.d / (1.d + Ntail * stail * stail / (score * score))
  psf = fcore * fermi_psf_base(x, score, gcore) + $
        (1 - fcore) * fermi_psf_base(x, stail, gtail)
  
  return, psf

end


; energy is in MeV
; dangle is degrees
function fermi_psf_pass8, class_, dangle, energy, silent=silent, eventtype=eventtype, pass7=pass7, irfs=irfs

  if keyword_set(pass7) then begin
     partitionval = 0d
  endif else begin

     if eventtype GE 1 and eventtype LE 2 then begin
        cuttype='FB'
        partitionval = eventtype -1 ;eventtype=1 corresponds to front, eventtype=2 corresponds to back
     endif
     if eventtype GE 3 and eventtype LE 6 then begin
        cuttype='PSF'
        partitionval = 6 - eventtype ;eventtype=3 corresponds to the best PSF, which is last in the file
     endif
     if eventtype GE 6 and eventtype LE 9 then begin
        cuttype='EDISP'
        partitionval = 9 - eventtype ;likewise, eventtype=6 will correspond to the best EDISP, which is last in the file
     endif
  endelse

;  psf_path =
;  '$FERMI_DIR/refdata/fermi/caldb/CALDB/data/glast/lat/bcf/psf/'
 psf_path = '/zfs/tslatyer/anaconda3/envs/fermi/share/fermitools/data/caldb/data/glast/lat/bcf/psf/'

  if keyword_set(pass7) then begin
     psf_file = 'psf_' + class_
  endif else begin
     if keyword_set(irfs) then begin
        case irfs of
           'p8r3': psf_file = 'psf_P8R3_' + class_ + '_V2'
           else: print, 'IRFs not recognized'
        endcase
     endif else begin
        psf_file = 'psf_P8R2_' + class_+'_V6'
     endelse
  endelse

  if keyword_set(cuttype) then psf_file += '_'+cuttype
  psf_file += '.fits'

  psf_path = psf_path + psf_file

  if ~ keyword_set(silent) then print, 'Reading PSF file: ', psf_path
  p  = mrdfits(psf_path, 1 + 3*partitionval, /silent)
  pp = mrdfits(psf_path, 2 + 3*partitionval, /silent, /no_tdim)

  c0 = pp.psfscale[0]
  c1 = pp.psfscale[1]
  bet = pp.psfscale[2]
  scalefac = sqrt((c0*(energy/100)^(bet))^2 + c1^2)
  x = dangle/!radeg/scalefac

; -------- read file
  
  ctheta = (p.ctheta_lo+p.ctheta_hi)/2.
  Ebin   = sqrt(p.energ_lo*p.energ_hi)

  Nenergy = n_elements(p.energ_lo) 
  grid = n_elements(energy) EQ 1
  Eind = interpol(lindgen(Nenergy), Ebin, energy)

  Nctheta = n_elements(p.ctheta_lo) 
  cind = interpol(lindgen(Nctheta), ctheta, $
                 0.9d*(1d + dblarr(n_elements(dangle)))) 
 ; cos(dangle/!radeg))
  ;cind = interpol(lindgen(Nctheta), ctheta, fltarr(n_elements(dangle)) + 1.)

  Ncore = interpolate(p.Ncore, Eind, cind, grid=grid)
  score = interpolate(p.score, Eind, cind, grid=grid)
  gcore = interpolate(p.gcore, Eind, cind, grid=grid)
  Ntail = interpolate(p.Ntail, Eind, cind, grid=grid)
  stail = interpolate(p.stail, Eind, cind, grid=grid)
  gtail = interpolate(p.gtail, Eind, cind, grid=grid)

  psf = reform(fermi_psf_with_tail(x, score, gcore, Ntail, stail, gtail))

    stop

  return, psf
end


; make plot of 68% and 95% containment radius as a 
;  function of energy.  Compare to 
;  http://www-glast.slac.stanford.edu/software/IS/glast_lat_performance.htm
;
pro fermi_psf_test

  x = (findgen(2500)+0.5)/100.

  energy = 10.d ^ (dindgen(41)/10+1.5)
  n = n_elements(energy) 


  front = [1, 1, 0, 0]
  back  = [0, 0, 1, 1]
  color = ['red', 'red', 'blue', 'blue']
  cfrac = [0.68, 0.95, 0.68, 0.95]
  line  = [0, 1, 0, 1]

  djs_plot, energy, xtit='Energy [MeV]', $
    ytit='containment angle [deg]', $
    /xlog, /ylog, chars=1.5, xr=[40, 250000], /xst, yr=[.1, 100]

  for k=0, n_elements(front)-1 do begin 
     xc = fltarr(n)
     for i=0, n-1 do begin
        psf = fermi_psf('P7REP_SOURCE_V15', x, energy[i], front=front[k], back=back[k])
        sum = total(psf*x, /cum)/total(psf*x)
        xc[i] = interpol(x, sum, cfrac[k])
     endfor

     djs_oplot, energy, xc, color=color[k], line=line[k]

  endfor

  return
end


