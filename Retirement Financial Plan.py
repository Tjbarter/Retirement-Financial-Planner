import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button
import matplotlib.gridspec as gridspec
from scipy.optimize import brentq
from datetime import date
from dateutil.relativedelta import relativedelta

# =====================
# INITIAL PARAMETERS
# =====================
birthday = date(1972, 12, 6)
oldest_retirement = 59
contribution_rate_annual = 60000
desired_floor_gross_annual = 64000
prior_income_first_year = 90000
horizon_age = 100

current_params = {
    'pot_me': 312500,
    'pot_spouse': 33000,
    'return_rate': 0.04,
    'floor_age': 75,
    'exp_rate': 2.0,        # Default exponential decay rate (alpha)
}

param_steps = {
    'pot_me': 10000,
    'pot_spouse': 5000,
    'return_rate': 0.005,
    'floor_age': 1,
    'exp_rate': 0.5,
}

param_formats = {
    'pot_me': '£{:,.0f}',
    'pot_spouse': '£{:,.0f}',
    'return_rate': '{:.1%}',
    'floor_age': '{:d}',
    'exp_rate': '{:.1f}',
}

other_pensions_annual = {
    'me': [
        {'name': 'war_dis',         'start_age': 52, 'end_age': 150, 'amount': 5000,  'taxable': False},
        {'name': 'military_early',  'start_age': 52, 'end_age': 55,  'amount': 20000, 'taxable': True},
        {'name': 'military_late',   'start_age': 55, 'end_age': 150, 'amount': 32000, 'taxable': True},
        {'name': 'state',           'start_age': 67, 'end_age': 150, 'amount': 12000, 'taxable': True},
    ],
    'spouse': [
        {'name': 'nhs',   'start_age': 67, 'end_age': 150, 'amount': 3000,  'taxable': True},
        {'name': 'state', 'start_age': 67, 'end_age': 150, 'amount': 12000, 'taxable': True},
    ]
}

# =====================
# TAX & DRAWDOWN HELPERS
# =====================

def get_tax_year(d):
    return d.year if d.month > 4 or (d.month == 4 and d.day >= 6) else d.year - 1


def _calculate_annual_tax(income):
    pa_base = 12570
    pa = pa_base
    if income > 100000:
        pa = max(0, pa_base - (income - 100000) / 2)
    tax = 0
    tax += max(0, min(income - pa, 37700)) * 0.20
    tax += max(0, min(income - (pa_base + 37700), 125140 - (pa_base + 37700))) * 0.40
    tax += max(0, income - 125140) * 0.45
    return tax


def compute_tax(monthly_taxable_income, prior_income_annual=0):
    annual_total = monthly_taxable_income * 12 + prior_income_annual
    return (_calculate_annual_tax(annual_total)
            - _calculate_annual_tax(prior_income_annual)) / 12


def get_guaranteed_pensions(age, pens):
    tot = 0
    for person in pens.values():
        for p in person:
            if p['start_age'] <= age < p.get('end_age', 999):
                tot += p['amount']
    return tot


def get_net_from_gross_monthly(gross, age, pens, split, prior_inc=0):
    guaranteed = get_guaranteed_pensions(age, pens)
    dc_draw = max(0, gross - guaranteed)
    total_tax = 0
    for who, p_list in pens.items():
        pr = prior_inc if who == 'me' else 0
        taxable_g = sum(
            p['amount'] for p in p_list
            if p['start_age'] <= age < p.get('end_age', 999) and p['taxable']
        )
        draw_for = dc_draw * (split if who == 'me' else 1 - split)
        total_tax += compute_tax(taxable_g + draw_for * 0.75, pr)
    return gross - total_tax


def find_gross_for_net_monthly(target_net, age, pens, split, prior_inc=0):
    args = (age, pens, split, prior_inc)
    try:
        return brentq(lambda g: get_net_from_gross_monthly(g, *args) - target_net,
                      target_net, target_net * 3)
    except ValueError:
        return brentq(lambda g: get_net_from_gross_monthly(g, *args) - target_net,
                      target_net, target_net * 6)


# Helper: build shaped path from peak->floor over Tm months

def shaped_path(peak, floor, Tm, shape, alpha):
    if Tm <= 0:
        return np.array([peak])
    t = np.arange(Tm, dtype=float)
    frac = t / Tm
    if shape == 'linear' or alpha is None:
        s = frac
    else:
        # normalised exponential easing 0->1; alpha>0 means faster early drop
        denom = (1 - np.exp(-alpha))
        if denom < 1e-9:
            s = frac
        else:
            s = (1 - np.exp(-alpha * frac)) / denom
    return peak - (peak - floor) * s


# =====================
# ROBUST PEAK SOLVERS (with infeasibility handling)
# =====================

def _solve_peak(pv_diff, low, high_start, max_tries=60):
    """
    Find value >= low s.t. pv_diff(value)=0. Returns (value, status) with status in:
      'ok', 'floor_infeasible', 'upper_capped', 'numeric_fail'
    """
    f_low = pv_diff(low)

    # If floor already needs more than the pot -> infeasible floor
    if f_low > 0:
        return low, 'floor_infeasible'

    high = max(high_start, low + 1e-6)
    f_high = None
    for _ in range(max_tries):
        try:
            f_high = pv_diff(high)
        except Exception:
            high *= 2
            continue
        if f_low * f_high <= 0:
            try:
                root = brentq(pv_diff, low, high, maxiter=200)
                return root, 'ok'
            except Exception:
                try:
                    root = brentq(pv_diff, max(low - 1e-9, 0), high * 1.000001, maxiter=200)
                    return root, 'ok'
                except Exception:
                    return high, 'numeric_fail'
        high *= 2
    return high, 'upper_capped'


def calculate_peak_net_monthly(ret_age, pot, floor_age, rr_annual, pens, split,
                               shape, alpha):
    rr_monthly = (1 + rr_annual) ** (1 / 12) - 1
    floor_net = get_net_from_gross_monthly(desired_floor_gross_annual / 12,
                                           floor_age, pens, split)
    start_date = birthday + relativedelta(years=int(ret_age),
                                          months=int((ret_age % 1) * 12))
    ret_tax_year = get_tax_year(start_date)
    Tm = int((floor_age - ret_age) * 12)
    if Tm <= 0:
        return floor_net, 'ok'

    def pv_diff(Pn):
        net_sched = shaped_path(Pn, floor_net, Tm, shape, alpha)
        gross_sched = []
        for i in range(Tm):
            d = start_date + relativedelta(months=i)
            a = (d - birthday).days / 365.25
            pr = prior_income_first_year if get_tax_year(d) == ret_tax_year else 0
            gross_sched.append(
                find_gross_for_net_monthly(net_sched[i], a, pens, split, pr)
            )
        gross_arr = np.array(gross_sched)
        ages = ret_age + np.arange(Tm) / 12
        guar = np.array([get_guaranteed_pensions(a, pens) for a in ages])
        draw = np.maximum(0, gross_arr - guar)
        return np.sum(draw * (1 + rr_monthly) ** (-np.arange(Tm))) - pot

    peak, status = _solve_peak(pv_diff, low=floor_net, high_start=floor_net * 1.5)
    return peak, status


def calculate_peak_gross_monthly(ret_age, pot, floor_age, rr_annual, pens,
                                 shape, alpha):
    rr_monthly = (1 + rr_annual) ** (1 / 12) - 1
    floor_m = desired_floor_gross_annual / 12
    Tm = int((floor_age - ret_age) * 12)
    if Tm <= 0:
        return floor_m, 'ok'

    ages = ret_age + np.arange(Tm) / 12
    guar = np.array([get_guaranteed_pensions(a, pens) for a in ages])

    def pv_diff(Pg):
        gross_sched = shaped_path(Pg, floor_m, Tm, shape, alpha)
        draw = np.maximum(0, gross_sched - guar)
        return np.sum(draw * (1 + rr_monthly) ** (-np.arange(Tm))) - pot

    peak, status = _solve_peak(pv_diff, low=floor_m, high_start=floor_m * 1.5)
    return peak, status


# =====================
# PLOTTING & UI
# =====================
fig = plt.figure(figsize=(12, 8))
# Control grid 2x5 to include Exp Rate control
gs_main = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.4)
ax = fig.add_subplot(gs_main[0])

gs_ctrl = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=gs_main[1],
                                           wspace=0.6, hspace=0.5)

axcolor = 'lightgoldenrodyellow'
value_displays = {}
active_widgets = []


def update(event):
    ax.clear()
    # read params
    pm0 = current_params['pot_me']
    ps0 = current_params['pot_spouse']
    rr0 = current_params['return_rate']
    fl0 = current_params['floor_age']
    alpha = current_params['exp_rate']

    today = date.today()
    cur_age = (today - birthday).days / 365.25
    contrib_m = contribution_rate_annual / 12

    # monthly pensions
    pens = {k: [{**p, 'amount': p['amount'] / 12} for p in v]
            for k, v in other_pensions_annual.items()}

    mode = s_radio.value_selected.lower()       # 'net'/'gross'
    shape = shape_radio.value_selected.lower()  # 'linear'/'exponential'
    ymode = yaxis_radio.value_selected          # 'Auto'/'From Zero'

    for ret_age in range(53, oldest_retirement + 1):
        if ret_age <= cur_age:
            continue

        # grow DC pots to retirement age
        months = int((ret_age - cur_age) * 12)
        pm, ps = pm0, ps0
        for _ in range(months):
            pm = pm * (1 + rr0 / 12) ** (1 / 12) + contrib_m
            ps = ps * (1 + rr0 / 12) ** (1 / 12)
        total = pm + ps
        split = pm / total if total > 0 else 0.5

        # find peak consistent with shape
        if mode == 'net':
            peak_m, status = calculate_peak_net_monthly(
                ret_age, total, fl0, rr0, pens, split, shape, alpha
            )
            floor_m = get_net_from_gross_monthly(
                desired_floor_gross_annual / 12, fl0, pens, split
            )
        else:
            peak_m, status = calculate_peak_gross_monthly(
                ret_age, total, fl0, rr0, pens, shape, alpha
            )
            floor_m = desired_floor_gross_annual / 12

        # Build plotted path over full horizon
        start_yr = peak_m * 12
        floor_yr = floor_m * 12
        Tyrs = fl0 - ret_age

        ages = np.arange(ret_age, horizon_age + 1, 1 / 12)
        rel_t = ages - ret_age

        if shape == 'linear' or Tyrs <= 0:
            frac = np.clip(rel_t / Tyrs, 0, 1) if Tyrs > 0 else 0
            income_m = peak_m - (peak_m - floor_m) * frac
        else:
            frac = np.clip(rel_t / Tyrs, 0, 1)
            denom = (1 - np.exp(-alpha))
            if denom < 1e-9:
                norm_exp = frac
            else:
                norm_exp = (1 - np.exp(-alpha * frac)) / denom
            income_m = peak_m - (peak_m - floor_m) * norm_exp

        decline = (start_yr - floor_yr) / Tyrs if Tyrs > 0 else 0

        # Styling/annotation based on solver status
        status_note = ""
        line_kwargs = {}
        if status == 'floor_infeasible':
            status_note = " (floor infeasible)"
            line_kwargs.update(dict(linestyle='--'))
        elif status in ('upper_capped', 'numeric_fail'):
            status_note = " (solver capped)"

        ax.plot(
            ages, income_m * 12,
            label=f'Retire {ret_age}   Δ£{decline:,.0f}/yr{status_note}',
            **line_kwargs
        )
        ax.scatter(ret_age, start_yr, s=30)
        ax.text(ret_age + 0.2, start_yr,
                f'£{start_yr:,.0f}', va='center', ha='left', fontsize=8)

    # draw floor line
    if mode == 'net':
        fl_line = get_net_from_gross_monthly(
            desired_floor_gross_annual / 12, fl0, pens, 0.5
        ) * 12
        suffix = '(Net)'
    else:
        fl_line = desired_floor_gross_annual
        suffix = '(Gross)'

    ax.axhline(fl_line, ls='--', color='gray',
               label=f'Floor @ {fl0} (£{fl_line:,.0f})')

    ax.set_title("Projected Retirement Income")
    ax.set_xlabel("Age")
    ax.set_ylabel(f"Annualised {suffix} Income (£)")
    ax.relim()
    ax.autoscale_view()
    if ymode == 'From Zero':
        ax.set_ylim(bottom=0)

    ax.legend(loc='upper right')
    ax.grid(True, which='both', ls='--', lw=0.5)

    # refresh param displays
    for k, btn in value_displays.items():
        btn.label.set_text(param_formats[k].format(current_params[k]))

    fig.canvas.draw_idle()


# Net/Gross toggle
ax_net = fig.add_subplot(gs_ctrl[0, 0], facecolor=axcolor)
s_radio = RadioButtons(ax_net, ('Net', 'Gross'), active=0)
s_radio.on_clicked(update)
active_widgets.append(s_radio)

# Shape toggle
ax_shape = fig.add_subplot(gs_ctrl[0, 1], facecolor=axcolor)
shape_radio = RadioButtons(ax_shape, ('Linear', 'Exponential'), active=0)
shape_radio.on_clicked(update)
active_widgets.append(shape_radio)

# Y-axis toggle
ax_yaxis = fig.add_subplot(gs_ctrl[0, 2], facecolor=axcolor)
yaxis_radio = RadioButtons(ax_yaxis, ('Auto', 'From Zero'), active=0)
yaxis_radio.on_clicked(update)
active_widgets.append(yaxis_radio)

# leave gs_ctrl[0,3] and [0,4] blank for spacing
fig.add_subplot(gs_ctrl[0, 3]).axis('off')
fig.add_subplot(gs_ctrl[0, 4]).axis('off')


# Param control factory (2×2 inside each cell)

def create_control_group(cell, label, key):
    igs = gridspec.GridSpecFromSubplotSpec(2, 2,
            subplot_spec=cell, wspace=0.1, hspace=0.1)
    axl = fig.add_subplot(igs[0, :])
    axl.text(0.5, 0.5, label, ha='center', va='center', fontsize=10)
    axl.axis('off')

    axv = fig.add_subplot(igs[1, 0])
    bval = Button(axv, '', color=axcolor, hovercolor='0.9')

    gs_btn = gridspec.GridSpecFromSubplotSpec(1, 2,
               subplot_spec=igs[1, 1], wspace=0.05)
    axm = fig.add_subplot(gs_btn[0, 0])
    bminus = Button(axm, '-', color=axcolor, hovercolor='0.9')
    axp = fig.add_subplot(gs_btn[0, 1])
    bplus = Button(axp, '+', color=axcolor, hovercolor='0.9')

    bminus.on_clicked(lambda e, k=key: adjust_param(k, -1))
    bplus.on_clicked(lambda e, k=key: adjust_param(k,  1))

    value_displays[key] = bval
    active_widgets.extend([bval, bminus, bplus])


def adjust_param(key, delta):
    current_params[key] += delta * param_steps[key]
    if 'pot' in key:
        current_params[key] = max(0, current_params[key])
    if 'return' in key:
        current_params[key] = min(max(0.005, current_params[key]), 0.1)
    if key == 'floor_age':
        current_params[key] = min(max(oldest_retirement + 1, current_params[key]), 90)
    if key == 'exp_rate':
        current_params[key] = min(max(0.1, current_params[key]), 10.0)
    update(None)


# Place parameter groups in row 1 (5 cells)
create_control_group(gs_ctrl[1, 0], "Your DC Pot",     'pot_me')
create_control_group(gs_ctrl[1, 1], "Spouse's DC Pot", 'pot_spouse')
create_control_group(gs_ctrl[1, 2], "Real Return",     'return_rate')
create_control_group(gs_ctrl[1, 3], "Floor Age",       'floor_age')
create_control_group(gs_ctrl[1, 4], "Exp Rate (α)",    'exp_rate')

# initial draw
update(None)
plt.show()
