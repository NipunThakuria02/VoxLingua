"""Seeded six-class controlled transformations with actual applied metadata."""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfilt, welch
from voxlingua.audio import quantize_int16
from voxlingua.features import frames, runs


def power(x): return float(np.mean(np.asarray(x,dtype=np.float64)**2))
def level(x): return 10*np.log10(max(power(x),1e-12))


def screen_reference(x: np.ndarray) -> list[str]:
    reasons=[]
    if np.mean(np.abs(x)>=0.999)>0.0005: reasons.append('reference_already_clipped')
    if any(a>0 and b<len(x) and b-a>=320 for a,b in runs(x==0)): reasons.append('reference_interior_zero_gap')
    if np.max(np.abs(x))<10**(-30/20): reasons.append('reference_peak_too_quiet')
    if level(x)<-45: reasons.append('reference_rms_too_quiet')
    return reasons


def _noise_excerpt(n: np.ndarray, length: int, rng) -> tuple[np.ndarray,int]:
    if len(n)>=length:
        offset=int(rng.integers(0,len(n)-length+1))
        return n[offset:offset+length].copy(),offset
    # Overlap-add 50-ms crossfades rather than a digital discontinuity at repeat boundaries.
    fade=min(800,len(n)//4)
    y=n.copy()
    while len(y)<length:
        ramp=np.linspace(0,1,fade,endpoint=False)
        blend=y[-fade:]*(1-ramp)+n[:fade]*ramp
        y=np.r_[y[:-fade],blend,n[fade:]]
    return y[:length],0


def derivative(x: np.ndarray, label: str, rng, noise: np.ndarray | None = None, *,
               target_snr_db: float | None = None) -> tuple[np.ndarray,dict]:
    x=np.asarray(x,dtype=np.float64)
    meta={}
    if label=='reference': return quantize_int16(x),meta
    if label=='added_noise':
        if noise is None: raise ValueError('A genuine declared noise source is required')
        n,offset=_noise_excerpt(np.asarray(noise,dtype=np.float64),len(x),rng)
        if level(n)<-60: raise ValueError('Noise excerpt too quiet')
        target=float(rng.uniform(0,20) if target_snr_db is None else target_snr_db)
        alpha=np.sqrt(power(x)/(power(n)*10**(target/10)))
        y=x+alpha*n
        gain=min(1,0.99/max(np.max(np.abs(y)),1e-12))
        y*=gain
        meta={'noise_offset_s':offset/16000,'applied_snr_db':float(10*np.log10(power(x)/(alpha**2*power(n)))), 'headroom_gain_db':float(20*np.log10(gain))}
    elif label=='clipping':
        fraction=float(np.exp(rng.uniform(np.log(0.005),np.log(0.08))))
        ceiling=float(quantize_int16(np.array([rng.uniform(0.8,0.9999)]))[0])
        q=np.quantile(np.abs(x),1-fraction)
        if q<1e-3: raise ValueError('Clipping input too quiet')
        gain=ceiling/q
        y=quantize_int16(np.clip(gain*x,-ceiling,ceiling))
        applied=float(np.mean(np.abs(y)>=ceiling-1e-6))
        if applied<0.002: raise ValueError('Insufficient clipping')
        return y,{'clip_ceiling':ceiling,'clip_gain_db':float(20*np.log10(gain)),'applied_clip_fraction':applied}
    elif label=='low_volume':
        target=float(rng.uniform(15,35))
        y=quantize_int16(x*10**(-target/20))
        applied=float(level(x)-level(y))
        if not 14.95<=applied<=35.05: raise ValueError('Attenuation out of tolerance')
        return y,{'applied_attenuation_db':applied}
    elif label=='frequency_filtering':
        kind=rng.choice(['lowpass','highpass','bandpass'],p=[0.4,0.2,0.4])
        logdraw=lambda lo,hi:float(np.exp(rng.uniform(np.log(lo),np.log(hi))))
        lo=hi=None
        if kind=='lowpass': hi=logdraw(1000,3500);cutoff=hi
        elif kind=='highpass': lo=logdraw(400,1500);cutoff=lo
        else: lo=logdraw(300,800);hi=logdraw(2000,3800);cutoff=[lo,hi]
        y=sosfilt(butter(4,cutoff,btype=kind,fs=16000,output='sos'),x)
        if np.max(np.abs(y))>0.999: raise ValueError('Filtering creates secondary clipping')
        y=quantize_int16(y)
        freq,px=welch(x,fs=16000,nperseg=512)
        _,py=welch(y,fs=16000,nperseg=512)
        masks=[]
        if lo: masks.append(freq<0.67*lo)
        if hi: masks.append(freq>1.5*hi)
        removed=[float(10*np.log10((px[m].sum()+1e-20)/(py[m].sum()+1e-20))) for m in masks]
        if not removed or min(removed)<3: raise ValueError('Insufficient stopband reduction')
        return y,{'filter_type':kind,'cutoff_lo_hz':lo,'cutoff_hi_hz':hi,'applied_stopband_energy_removed_db':min(removed),'applied_rms_change_db':float(level(y)-level(x))}
    elif label=='interruption':
        count=int(rng.integers(1,6));gaps=[];y=x.copy()
        med=np.median(10*np.log10(np.maximum(np.mean(frames(x,400,160)**2,axis=1),1e-12)))
        for _ in range(count):
            duration=int(np.exp(rng.uniform(np.log(480),np.log(4800))))
            if sum(b-a for a,b,_,_ in gaps)+duration>0.25*len(x): raise ValueError('Too many gaps')
            found=False
            for _ in range(40):
                start=int(rng.integers(4800,len(x)-4800-duration));end=start+duration
                if any(not (end+1600<=a or start>=b+1600) for a,b,_,_ in gaps): continue
                if level(x[start:end])<med-6: continue
                mode=str(rng.choice(['zero','attenuate','noise'],p=[0.3,0.35,0.35]))
                if mode=='zero': fill=np.zeros(duration)
                elif mode=='attenuate': fill=x[start:end]*0.001
                else:
                    floor=np.quantile(np.sqrt(np.mean(frames(x,400,160)**2,axis=1)),0.1)
                    fill=rng.normal(0,floor,duration)
                fade=int(rng.integers(0,81))
                if fade:
                    fade=min(fade,duration//2);ramp=np.linspace(0,1,fade,endpoint=False)
                    fill[:fade]=x[start:start+fade]*(1-ramp)+fill[:fade]*ramp
                    fill[-fade:]=fill[-fade:]*(1-ramp)+x[end-fade:end]*ramp
                y[start:end]=fill;gaps.append((start,end,mode,fade));found=True;break
            if not found: raise ValueError('No valid speech gap window')
        y=quantize_int16(y)
        original_energy=sum(np.sum(x[a:b]**2) for a,b,_,_ in gaps)
        new_energy=sum(np.sum(y[a:b].astype(np.float64)**2) for a,b,_,_ in gaps)
        removed=float(10*np.log10(original_energy/max(new_energy,1e-20)))
        if removed<6: raise ValueError('Gap did not remove enough energy')
        return y,{'n_gaps':len(gaps),'gap_starts_s':[a/16000 for a,_,_,_ in gaps],'gap_durations_ms':[(b-a)/16 for a,b,_,_ in gaps],'gap_fill_modes':[mode for _,_,mode,_ in gaps],'gap_energy_removed_db':removed}
    else: raise ValueError('Unknown class')
    return quantize_int16(y),meta
