from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

out = Path('/home/dshen/Code/drift-with-trailer/output/pdf/closed-loop-reverse-control.pdf')
out.parent.mkdir(parents=True, exist_ok=True)
font_dir = Path('/usr/share/fonts/liberation-sans-fonts')
pdfmetrics.registerFont(TTFont('Sans', str(font_dir / 'LiberationSans-Regular.ttf')))
pdfmetrics.registerFont(TTFont('SansBold', str(font_dir / 'LiberationSans-Bold.ttf')))
W,H = 1008,792  # 14 x 11 inches
c = canvas.Canvas(str(out), pagesize=(W,H))
c.setTitle('Closed-loop reverse tractor-trailer control')
navy='#00274C'; teal='#008C88'; gray='#879BA9'; ink='#142A3C'; muted='#5F6F7B'; pale='#EDF3F7'; grid='#DCE5EA'; white='#FFFFFF'
def fill(hex): c.setFillColor(HexColor(hex))
def stroke(hex): c.setStrokeColor(HexColor(hex))
def box(x,y,w,h,color):
    fill(color); c.rect(x,H-y-h,w,h,fill=1,stroke=0)
def label(s,x,y,size=13,color=ink,bold=False):
    fill(color);c.setFont('SansBold' if bold else 'Sans',size);c.drawString(x,H-y-size,s)
def right(s,x,y,size=12,color=ink,bold=False):
    fill(color);c.setFont('SansBold' if bold else 'Sans',size);c.drawRightString(x,H-y-size,s)
def line(x1,y1,x2,y2,color=grid,width=1):
    stroke(color);c.setLineWidth(width);c.line(x1,H-y1,x2,H-y2)
def circle(x,y,r,color):
    fill(color);c.circle(x,H-y,r,fill=1,stroke=0)

envs=['Asphalt','Snow','Slope','Snow + Slope']
learn_comp=[18,18,16,10]
prior_comp=[11,10,14,6]
# Each environment contributes two reverse setpoints, -30 then -50 km/h.
learn_speed=[29.1,47.2,29.1,46.0,25.4,42.1,20.4,30.4]
prior_speed=[19.7,22.7,19.6,22.8,22.3,22.5,16.7,18.1]
learn_hitch=[1.14,1.75,1.18,1.82,1.03,3.06,10.82,15.41]
prior_hitch=[5.25,6.70,7.82,8.49,2.86,6.48,11.47,15.86]
targets=[30,50]*4
learn_error=[round(t-v,1) for t,v in zip(targets,learn_speed)]
prior_error=[round(t-v,1) for t,v in zip(targets,prior_speed)]
assert all(a<b for a,b in zip(learn_error,prior_error))
assert all(a<b for a,b in zip(learn_hitch,prior_hitch))
assert sum(learn_comp)==62 and sum(prior_comp)==41

box(0,0,W,H,white)
box(0,0,W,7,navy)
label('Closed-loop reverse control',32,24,27,navy,True)
label('Learned-dynamics MPPI vs. Fiala analytic-prior MPPI',32,65,15,muted)
# Shared legend
circle(693,67,6,teal);label('Learned',706,56,13,ink,True)
circle(832,67,6,gray);label('Analytic prior',845,56,13,ink,True)

label('Runs completed',32,107,18,navy,True)
label('Two reverse targets per environment; 18 runs possible',32,132,11,muted)
left=192;span=626
for i,env in enumerate(envs):
    cy=178+i*37
    label(env,32,cy-11,13,ink,True)
    box(left,cy-12,span,10,pale)
    box(left,cy+3,span,10,pale)
    box(left,cy-12,span*learn_comp[i]/18,10,teal)
    box(left,cy+3,span*prior_comp[i]/18,10,gray)
    label(f'{learn_comp[i]}/18',835,cy-17,12,teal,True)
    label(f'{prior_comp[i]}/18',902,cy-2,12,muted,True)
# clarify labels at far right with separate header
label('Learned',824,141,10,teal,True)
label('Prior',899,141,10,muted,True)
line(32,326,976,326,'#B6C5CF',1)

label('Per-setpoint performance',32,343,18,navy,True)
label('Lower values are better; each marker averages all nine runs, including unsuccessful runs.',32,368,11,muted)
label('Target',123,407,11,muted,True)
label('Mean speed shortfall (km/h)',205,405,15,navy,True)
label('RMS hitch angle (degrees)',608,405,15,navy,True)
# Panel bounds. Use identical row positions to make each environment easy to scan.
x1a,x1b=207,526
x2a,x2b=604,938
max1,max2=35,18
for tick in [0,10,20,30]:
    x=x1a+(x1b-x1a)*tick/max1
    line(x,438,x,708,grid,.8); right(str(tick),x+4,715,10,muted)
for tick in [0,5,10,15]:
    x=x2a+(x2b-x2a)*tick/max2
    line(x,438,x,708,grid,.8); right(str(tick),x+4,715,10,muted)
for i in range(8):
    y=455+i*34
    if i%2==0:
        box(32,y-13,923,68,'#F8FAFB')
        label(envs[i//2],34,y-10,12,ink,True)
    label(f'−{targets[i]}',137,y-8,12,muted)
    for xa,xb,maximum,lv,pv in [(x1a,x1b,max1,learn_error[i],prior_error[i]),(x2a,x2b,max2,learn_hitch[i],prior_hitch[i])]:
        p=xa+(xb-xa)*pv/maximum
        l=xa+(xb-xa)*lv/maximum
        line(l,y,p,y,'#AAB9C2',2)
        circle(p,y,5.3,gray)
        circle(l,y,6.0,teal)
    if i%2==1 and i<7:
        line(32,y+17,955,y+17,'#CBD7DE',1)
line(32,745,976,745,'#B6C5CF',1)
label('Completion: 62/72 learned vs. 41/72 prior reverse runs.',32,752,12,navy,True)
right('Shortfall = |target| − mean speed. Source: Table II.',976,752,11,muted)
c.showPage();c.save()
print(out)
