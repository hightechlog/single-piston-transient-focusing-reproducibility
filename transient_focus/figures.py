"""Generate the eight figures reproduced from the published numerical model.

Physical calculations are kept in model.py, design.py, and propagation.py. This
module contains only plotting and display normalization.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from .model import aperture_response, aperture_signature, differential_delay
from .parameters import (
    BANDWIDTH_HZ, CENTER_FREQUENCY_HZ, CUT_MAX_DB, CUT_MIN_DB,
    FIELD_MAX_DB, FIELD_MIN_DB, NOMINAL_TARGET_M, PISTON_RADIUS_M,
    SOUND_SPEED_M_S,
)
from .propagation import real_waveform

METHODS_DESIGN = ("best_delay", "phase", "mf")
METHODS_TRANSIENT = ("plain", "best_delay", "phase", "mf")
METHODS_DISPLAY = ("best_delay", "phase", "mf")
TITLES = {
    "plain": "Geometric-delay broadband", "best_delay": "Best scalar delay",
    "phase": "Phase-only", "mf": "Matched waveform", "cw": "CW at $f_0$",
}
STYLES = {
    "plain": dict(color="black", ls="-", lw=1.5),
    "best_delay": dict(color="black", ls="-.", lw=1.5),
    "phase": dict(color="black", ls="--", lw=1.2),
    "mf": dict(color="black", ls=":", lw=1.2),
    "cw": dict(color="0.45", ls="-", lw=1.0),
}


def set_style() -> None:
    """Apply the common monochrome publication style."""
    plt.rcParams.update({
        "font.family":"sans-serif", "font.size":9.0, "axes.labelsize":9.0,
        "axes.titlesize":10.0, "legend.fontsize":8.0, "xtick.labelsize":8.0,
        "ytick.labelsize":8.0, "axes.linewidth":0.8, "lines.linewidth":1.25,
        "savefig.bbox":"tight", "pdf.fonttype":42, "ps.fonttype":42,
    })


def _save(fig: plt.Figure, out: Path, number: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    stem = f"figure_{number:02d}"
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _label(ax: plt.Axes, text: str, x: float = -0.11) -> None:
    ax.text(x, 1.035, text, transform=ax.transAxes, fontweight="bold",
            va="bottom", clip_on=False)


def _db(x: np.ndarray, ref: float | None = None) -> np.ndarray:
    eps = np.finfo(float).tiny
    x = np.asarray(x, float)
    ref = max(float(np.max(x) if ref is None else ref), eps)
    return 20 * np.log10(np.maximum(x, eps) / ref)


def _mirror(field: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (np.concatenate((field[:0:-1, ...], field), axis=0),
            np.concatenate((-r[:0:-1], r)))


def _contours(ax: plt.Axes, x: np.ndarray, y: np.ndarray, db: np.ndarray) -> None:
    ax.contour(x, y, db, levels=(-6., -3.), colors="black",
               linestyles=("--", "-"), linewidths=.8, zorder=3)


def _rz(ax: plt.Axes, amp: np.ndarray, r: np.ndarray, z: np.ndarray,
        target: float | None, title: str, label: str, label_x: float = -0.11) -> plt.AxesImage:
    db, rfull = _mirror(_db(amp), r)
    im = ax.imshow(db.T, origin="lower", aspect="auto",
                   extent=[rfull[0]*1e3, rfull[-1]*1e3, z[0]*1e3, z[-1]*1e3],
                   cmap="gray", vmin=FIELD_MIN_DB, vmax=FIELD_MAX_DB)
    _contours(ax, rfull*1e3, z*1e3, db.T)
    ip = np.unravel_index(int(np.argmax(amp)), amp.shape)
    rp, zp = float(r[ip[0]]*1e3), float(z[ip[1]]*1e3)
    ax.plot(rp, zp, "x", color="black", ms=5, mew=1.1)
    if rp > 0:
        ax.plot(-rp, zp, "x", color="black", ms=5, mew=1.1)
    if target is not None:
        ax.plot(0, target*1e3, "o", ms=3.3, mfc="none", mec="black")
        ax.axhline(target*1e3, color="black", ls="--", lw=.7)
    ax.axvline(0, color="black", ls=":", lw=.7)
    ax.set(xlabel="r (mm)", ylabel="z (mm)", title=title)
    _label(ax, label, x=label_x)
    return im


def figure_01(out: Path) -> None:
    """Geometry, aperture delay, and residual direct-edge response."""
    a, c, f0, b = PISTON_RADIUS_M, SOUND_SPEED_M_S, CENTER_FREQUENCY_HZ, BANDWIDTH_HZ
    marks = (.020, .050, .150, .250)
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True)
    z0 = .10
    ax[0].plot([0,0], [-a,a], color="black", lw=5, solid_capstyle="butt")
    ax[0].plot(z0, 0, "o", ms=4, mfc="white", mec="black")
    ax[0].plot([0,z0], [0,0], color=".35", ls="--", lw=1)
    ax[0].plot([0,z0], [a,0], color="black", lw=1)
    ax[0].text(z0, .007, r"$z_0$", ha="center")
    ax[0].text(.45*z0, .004, r"$z$", ha="center")
    ax[0].text(.47*z0, .57*a, r"$\sqrt{z^2+a^2}$", rotation=-22, ha="center")
    rx = -.012
    ax[0].annotate("", xy=(rx,a), xytext=(rx,0), arrowprops=dict(arrowstyle="<->", lw=.8))
    ax[0].text(rx-.004, a/2, r"$a$", ha="right", va="center")
    ax[0].set(xlim=(-.025,.12), ylim=(-.06,.065), title=r"$r$--$z$ geometry")
    ax[0].set_aspect("equal"); ax[0].axis("off"); _label(ax[0], "(a)")

    z = np.linspace(.005, .300, 500)
    ax[1].plot(z*1e3, differential_delay(z,a,c)*1e6, color="black")
    for zm in marks:
        ax[1].plot(zm*1e3, float(differential_delay(zm,a,c))*1e6,
                   "o", ms=3.5, mfc="white", mec="black")
    ax[1].set(xlabel="Axial distance z (mm)", ylabel=r"$\Delta\tau$ ($\mu$s)",
              title="Differential aperture delay"); _label(ax[1], "(b)")

    f = np.linspace(f0-b/2, f0+b/2, 900)
    for zm, ls in zip(marks, ("-","--","-.",":")):
        ax[2].plot(f/1e3, np.abs(aperture_response(f,zm,a,c)[0]),
                   color="black", ls=ls, label=f"{zm*1e3:.0f} mm")
    ax[2].set(xlabel="Frequency (kHz)", ylabel=r"$|D(z,f)|$",
              title="Direct--edge interference")
    ax[2].legend(loc="lower right", fontsize=8); _label(ax[2], "(c)")
    _save(fig, out, 1)


def figure_02(out: Path) -> None:
    """Nominal q-eta interpretation and in-band interference zeros."""
    tau = float(differential_delay(NOMINAL_TARGET_M, PISTON_RADIUS_M, SOUND_SPEED_M_S))
    df0 = 1/tau; q = CENTER_FREQUENCY_HZ*tau; eta = BANDWIDTH_HZ*tau
    fmin = CENTER_FREQUENCY_HZ-BANDWIDTH_HZ/2; fmax = CENTER_FREQUENCY_HZ+BANDWIDTH_HZ/2
    f = np.linspace(0, 230e3, 2400)
    s = aperture_signature(f, NOMINAL_TARGET_M, PISTON_RADIUS_M, SOUND_SPEED_M_S)[0]
    zeros = np.arange(1,4)*df0
    fig, ax = plt.subplots(figsize=(7.2,3.5))
    ax.axvspan(fmin/1e3, fmax/1e3, color=".88", label="Operating band: 50–150 kHz")
    ax.plot(f/1e3, s, color="black", label=r"$S(z_0,f)=\sin[\pi f\Delta\tau(z_0)]$")
    ax.axhline(0, color="black", lw=.8)
    for x in (fmin,fmax): ax.axvline(x/1e3, color="black", ls="--", lw=.9)
    ax.axvline(CENTER_FREQUENCY_HZ/1e3, color="black", lw=1.4)
    for x in zeros: ax.axvline(x/1e3, color="black", ls=":", lw=1)
    ax.plot(zeros[:2]/1e3, [0,0], "o", ms=8, mfc="none", mec="black")
    ax.text(zeros[0]/1e3+6, 0, r"$f_1$", ha="right", va="bottom")
    ax.text(zeros[1]/1e3-4, 0, r"$f_2$", ha="right", va="bottom")
    ax.text(zeros[0]/1e3+16, .12, rf"${zeros[0]/1e3:.1f}\,\mathrm{{kHz}}$", ha="center", va="top", fontsize=8)
    ax.text(zeros[1]/1e3+18, .12, rf"${zeros[1]/1e3:.1f}\,\mathrm{{kHz}}$", ha="center", va="top", fontsize=8)
    ax.annotate("", xy=(zeros[1]/1e3,.6), xytext=(zeros[0]/1e3,.6), arrowprops=dict(arrowstyle="<->"))
    ax.text((zeros[0]+zeros[1])/2e3+14, .655, rf"$\Delta f_{{\rm zero}}={df0/1e3:.2f}\,\mathrm{{kHz}}$", ha="center")
    ax.annotate("", xy=(fmax/1e3,1), xytext=(fmin/1e3,1), arrowprops=dict(arrowstyle="<->"))
    ax.text((fmin+fmax)/2e3-2, 1.055, rf"$B=\eta\,\Delta f_{{\rm zero}} \quad \eta={eta:.2f}$", ha="center")
    ax.text(CENTER_FREQUENCY_HZ/1e3+16, -1.28, rf"$f_0={CENTER_FREQUENCY_HZ/1e3:.0f}\,\mathrm{{kHz}}$", ha="center")
    ax.text(CENTER_FREQUENCY_HZ/1e3+2.8, -1.32, rf"$q=f_0/\Delta f_{{\rm zero}}={q:.2f}$", ha="left", va="top")
    ax.set(xlim=(0,230), ylim=(-1.52,1.30), xlabel=r"Frequency $f$ (kHz)", ylabel=r"$S(z_0,f)$")
    ax.set_yticks([-1,0,1]); ax.grid(alpha=.18); ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout(); _save(fig, out, 2)


def figure_03(out: Path, r: np.ndarray, z: np.ndarray, t: np.ndarray,
              snap: dict[str,np.ndarray], zt: dict[str,np.ndarray]) -> None:
    """Nominal instantaneous r-z fields and on-axis z-t envelopes."""
    fig, axes = plt.subplots(2,3,figsize=(7.2,6.2),constrained_layout=True); im = None
    for j,n in enumerate(METHODS_DESIGN):
        im = _rz(axes[0,j], snap[n], r, z, NOMINAL_TARGET_M, TITLES[n], f"({chr(97+j)})")
    for j,n in enumerate(METHODS_DESIGN):
        db = _db(zt[n]); a = axes[1,j]
        im = a.imshow(db, origin="lower", aspect="auto",
                      extent=[t[0]*1e6,t[-1]*1e6,z[0]*1e3,z[-1]*1e3],
                      cmap="gray", vmin=FIELD_MIN_DB, vmax=FIELD_MAX_DB)
        _contours(a,t*1e6,z*1e3,db)
        ip = np.unravel_index(int(np.argmax(zt[n])),zt[n].shape)
        a.plot(t[ip[1]]*1e6,z[ip[0]]*1e3,"x",color="black",ms=5)
        a.plot(0,NOMINAL_TARGET_M*1e3,"o",ms=3.3,mfc="none",mec="black")
        a.axhline(NOMINAL_TARGET_M*1e3,color="black",ls="--",lw=.7); a.axvline(0,color="black",ls=":",lw=.7)
        a.set(xlabel=r"$t$ ($\mu$s)",ylabel="z (mm)",title=TITLES[n]); _label(a,f"({chr(100+j)})")
    if im is not None: fig.colorbar(im,ax=axes,fraction=.025,pad=.02)
    _save(fig,out,3)


def figure_04(out: Path, r: np.ndarray, z: np.ndarray, cw: np.ndarray,
              peaks: dict[str,np.ndarray]) -> None:
    """Nominal pointwise peak-over-time maps."""
    fig,axes=plt.subplots(2,2,figsize=(7.2,6.1),constrained_layout=True); im=None
    for j,n in enumerate(("cw","best_delay","phase","mf")):
        im=_rz(axes.flat[j],cw if n=="cw" else peaks[n],r,z,NOMINAL_TARGET_M,TITLES[n],f"({chr(97+j)})")
    if im is not None: fig.colorbar(im,ax=axes,fraction=.025,pad=.02)
    _save(fig,out,4)


def figure_05(out: Path, r: np.ndarray, z: np.ndarray, snap: dict[str,np.ndarray],
              cw: np.ndarray, peaks: dict[str,np.ndarray]) -> None:
    """Lateral and axial cuts for instantaneous and peak fields."""
    iz=int(np.argmin(np.abs(z-NOMINAL_TARGET_M))); fig,axes=plt.subplots(2,2,figsize=(7.2,5.6),constrained_layout=True)
    for row,fields in enumerate((snap,peaks)):
        p,rf=_mirror(_db(cw[:,iz]),r); axes[row,0].plot(rf*1e3,p,label=TITLES["cw"],**STYLES["cw"])
        for n in METHODS_DISPLAY:
            p,rf=_mirror(_db(fields[n][:,iz]),r); axes[row,0].plot(rf*1e3,p,label=TITLES[n],**STYLES[n])
        axes[row,1].plot(z*1e3,_db(cw[0]),label=TITLES["cw"],**STYLES["cw"])
        for n in METHODS_DISPLAY: axes[row,1].plot(z*1e3,_db(fields[n][0]),label=TITLES[n],**STYLES[n])
    titles=(r"Lateral cut at $z=z_0$, $t=0$",r"Axial cut at $t=0$","Lateral cut of peak field","Axial cut of peak field")
    for i,a in enumerate(axes.flat):
        a.axhline(-3,ls="--",lw=.75); a.axhline(-6,ls=":",lw=.75)
        a.axvline(0 if i in (0,2) else NOMINAL_TARGET_M*1e3,ls=":" if i in (0,2) else "-.",lw=.75)
        a.set(xlabel="r (mm)" if i in (0,2) else "z (mm)",ylabel="Level relative to curve maximum (dB)",title=titles[i],ylim=(CUT_MIN_DB,CUT_MAX_DB))
        a.grid(alpha=.25); a.autoscale(enable=True,axis="x",tight=True); _label(a,f"({chr(97+i)})")
    h,l=axes[1,1].get_legend_handles_labels(); fig.legend(h,l,loc="lower center",bbox_to_anchor=(.5,-.085),ncol=4,frameon=False,fontsize=8)
    _save(fig,out,5)


def _phase(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    p=np.unwrap(np.angle(x)); p-=p[int(np.argmin(np.abs(f-CENTER_FREQUENCY_HZ)))]; return p/np.pi


def figure_06(out: Path, h0: np.ndarray, f: np.ndarray, spectra: dict[str,np.ndarray],
              window_us: float, n_time: int) -> None:
    """Excitation waveforms/spectra and corresponding target response."""
    t=np.linspace(-window_us*1e-6,window_us*1e-6,n_time); deramp=np.exp(-1j*2*np.pi*f*NOMINAL_TARGET_M/SOUND_SPEED_M_S)
    residual={n:spectra[n]*deramp for n in METHODS_TRANSIENT}; drive={n:real_waveform(residual[n],f,t) for n in METHODS_TRANSIENT}
    ps={n:h0*spectra[n] for n in METHODS_TRANSIENT}; pressure={n:real_waveform(ps[n],f,t) for n in METHODS_TRANSIENT}
    dr=max(np.max(np.abs(x)) for x in drive.values()); ds=max(np.max(np.abs(x)) for x in residual.values())
    pr=max(np.max(np.abs(x)) for x in pressure.values()); psr=max(np.max(np.abs(x)) for x in ps.values())
    fig,a=plt.subplots(2,3,figsize=(7.2,5.8),constrained_layout=True)
    for n in METHODS_DISPLAY:
        a[0,0].plot(t*1e6,drive[n]/dr,label=TITLES[n],**STYLES[n]); a[0,1].plot(f*1e-3,np.abs(residual[n])/ds,**STYLES[n])
        ph=_phase(residual[n],f); ph[np.abs(residual[n])<1e-3*np.max(np.abs(residual[n]))]=np.nan; a[0,2].plot(f*1e-3,ph,**STYLES[n])
        a[1,0].plot(t*1e6,pressure[n]/pr,**STYLES[n]); a[1,1].plot(f*1e-3,np.abs(ps[n])/psr,**STYLES[n])
        ph=_phase(ps[n],f); ph[np.abs(ps[n])<1e-3*np.max(np.abs(ps[n]))]=np.nan; a[1,2].plot(f*1e-3,ph,**STYLES[n])
    ic=int(np.argmin(np.abs(f-CENTER_FREQUENCY_HZ))); hc=h0[ic]
    a[0,0].plot(t*1e6,np.cos(2*np.pi*CENTER_FREQUENCY_HZ*t),label=TITLES["cw"],**STYLES["cw"])
    scale=abs(hc)*dr/pr; phase=np.angle(hc*np.exp(1j*2*np.pi*CENTER_FREQUENCY_HZ*NOMINAL_TARGET_M/SOUND_SPEED_M_S))
    a[1,0].plot(t*1e6,scale*np.cos(2*np.pi*CENTER_FREQUENCY_HZ*t+phase),**STYLES["cw"])
    for x in (a[0,1],a[0,2],a[1,1],a[1,2]): x.axvline(CENTER_FREQUENCY_HZ*1e-3,**STYLES["cw"])
    settings=((r"Relative time ($\mu$s)","Drive / common peak","Real excitation"),("Frequency (kHz)",r"$|V(f)|$ / common peak","Excitation magnitude"),("Frequency (kHz)",r"Residual phase / $\pi$","Excitation phase"),(r"$t$ ($\mu$s)","Pressure / common peak","Target pressure"),("Frequency (kHz)",r"$|P_0(f)|$ / common peak","Target magnitude"),("Frequency (kHz)",r"Phase / $\pi$","Target phase"))
    for i,x in enumerate(a.flat):
        x.set(xlabel=settings[i][0],ylabel=settings[i][1],title=settings[i][2]); x.grid(alpha=.25); x.autoscale(enable=True,axis="both",tight=True); x.margins(y=.03); _label(x,f"({chr(97+i)})")
    h,l=a[0,0].get_legend_handles_labels(); fig.legend(h,l,loc="lower center",bbox_to_anchor=(.5,-.065),ncol=4,frameon=False,fontsize=8)
    _save(fig,out,6)


def _retarget(out: Path, number: int, r: np.ndarray, z: np.ndarray, cw: np.ndarray,
              broad: np.ndarray, matched: dict[float,np.ndarray]) -> None:
    panels=(("CW",cw,None),("Geometric-delay\nbroadband",broad,NOMINAL_TARGET_M),(r"$z_0=20$ mm",matched[.020],.020),(r"$z_0=50$ mm",matched[.050],.050),(r"$z_0=100$ mm",matched[.100],.100),(r"$z_0=150$ mm",matched[.150],.150))
    fig,a=plt.subplots(2,3,figsize=(7.2,6),constrained_layout=True); im=None
    for i,(title,amp,target) in enumerate(panels):
        im=_rz(a.flat[i],amp,r,z,target,title,f"({chr(97+i)})",label_x=-.16)
    if im is not None: fig.colorbar(im,ax=a,fraction=.025,pad=.02)
    _save(fig,out,number)


def figure_07(out: Path, r: np.ndarray, z: np.ndarray, cw: np.ndarray,
              broad: np.ndarray, matched: dict[float,np.ndarray]) -> None:
    """Independent-target instantaneous envelopes at the common design time t=0."""
    _retarget(out,7,r,z,cw,broad,matched)


def figure_08(out: Path, r: np.ndarray, z: np.ndarray, cw: np.ndarray,
              broad: np.ndarray, matched: dict[float,np.ndarray]) -> None:
    """Independent-target pointwise peak-over-time envelopes."""
    _retarget(out,8,r,z,cw,broad,matched)
