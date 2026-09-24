from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image
from pathlib import Path
root=Path('/home/dshen/Code/drift-with-trailer')
for name,file in [('Regular','LiberationSans-Regular.ttf'),('Bold','LiberationSans-Bold.ttf')]:
 pdfmetrics.registerFont(TTFont(name,'/usr/share/fonts/liberation-sans-fonts/'+file))
W,H=3456,2592
c=canvas.Canvas(str(root/'output/pdf/tractor-trailer-poster-iros-layout.pdf'),pagesize=(W,H))
c.setTitle('Learning Dynamics for High-Speed Tractor-Trailer Control | 48 x 36 inch poster mockup')
navy='#00274C'; gold='#FFCB05'; ink='#142A3C'; gray='#52616E'; pale='#EDF3F7'; teal='#008C88'
def rect(x,y,w,h,color):
 c.setFillColor(HexColor(color));c.rect(x,H-y-h,w,h,stroke=0,fill=1)
def text(s,x,y,size=30,font='Regular',color=ink):
 c.setFillColor(HexColor(color));c.setFont(font,size);c.drawString(x,H-y-size,s)
def para(s,x,y,w,size=30,color=ink,bold=False):
 p=Paragraph(s,ParagraphStyle('p',fontName='Bold' if bold else 'Regular',fontSize=size,leading=size*1.27,textColor=HexColor(color)))
 _,h=p.wrap(w,2000);p.drawOn(c,x,H-y-h);return h

def heading(n,s,x,y,w):
 text(n,x,y,27,'Bold',teal);text(s,x+55,y-5,38,'Bold',navy);rect(x,y+55,w,3,'#CCD8E0')
def bullets(items,x,y,w,size=31):
 for s in items:
  rect(x,y+13,8,8,teal);h=para(s,x+26,y,w-26,size);y+=h+22
 return y

def crop(page,box,name):
 im=Image.open(root/f'tmp/pdfs/source-{page}.png');fac=im.height/1400
 out=root/f'tmp/pdfs/{name}.png';im.crop(tuple(round(a*fac) for a in box)).save(out);return out

def pic(path,x,y,w,h):
 im=Image.open(path); iw,ih=im.size;scale=min(w/iw,h/ih);dw,dh=iw*scale,ih*scale
 c.drawImage(str(path),x+(w-dw)/2,H-y-(h+dh)/2,dw,dh,mask='auto')
def center(s,x,y,w,size=34,font='Bold',color=ink):
 sw=pdfmetrics.stringWidth(s,font,size);text(s,x+(w-sw)/2,y,size,font,color)
def bar(s,x,y,w):
 rect(x,y,w,78,navy);center(s,x,y+12,w,43,color='#FFFFFF')
def line(x1,y1,x2,y2,color=gray,width=3):
 c.setStrokeColor(HexColor(color));c.setLineWidth(width);c.line(x1,H-y1,x2,H-y2)
def arrow(x1,y1,x2,y2):
 line(x1,y1,x2,y2)
 if y2>y1:line(x2,y2,x2-10,y2-16);line(x2,y2,x2+10,y2-16)
 elif x2>x1:line(x2,y2,x2-16,y2-10);line(x2,y2,x2-16,y2+10)
 elif x2<x1:line(x2,y2,x2+16,y2-10);line(x2,y2,x2+16,y2+10)
def node(title,sub,x,y,w,h=114):
 rect(x,y,w,h,pale);center(title,x,y+18,w,33);center(sub,x,y+66,w,26,'Regular',gray)
geom=crop(2,(561,86,971,304),'geometry')
track=crop(6,(130,90,1007,469),'track')
forward=crop(7,(95,87,533,310),'forward-rollouts')
photos=[crop(6,box,n) for box,n in [((100,570,313,727),'asphalt'),((323,570,537,727),'snow'),((545,570,759,727),'slope'),((772,570,982,727),'snow-slope')]]
L,C,R=72,1194,2316;cw=1068
rect(0,0,W,H,'#FFFFFF')
# Compact academic title and author block
rect(72,58,390,12,gold)
text('UNIVERSITY OF',72,95,30,'Bold',navy);text('MICHIGAN',72,135,53,'Bold',navy)
center('Learning Dynamics for High-Speed',510,55,2470,79,color=navy)
center('Tractor-Trailer Control',510,146,2470,79,color=navy)
center('Aaron Chen*  ·  David Shen*  ·  Taekyung Kim*  ·  Kaleb Ben Naveed  ·  Dimitra Panagou',500,251,2460,33,'Regular')
center('University of Michigan, Ann Arbor, USA  |  EECS · Robotics · Aerospace Engineering',500,300,2460,29,'Regular',gray)
c.setStrokeColor(HexColor('#B6C3CC'));c.setLineWidth(2);c.rect(3156,H-270,190,190,fill=0,stroke=1)
center('VIDEO / CODE',3156,129,190,23);center('QR placeholder',3156,170,190,21,'Regular',gray)
rect(72,371,3312,6,gold)
# left
bar('Introduction & Motivation',L,420,cw)
para('<b>Model-based control depends on accurate dynamics.</b>',L+12,532,cw-24,34)
for y,title,body in [
 (620,'Opportunity — Learn what analytic models miss','Learned dynamics can improve prediction when system properties and environmental conditions are uncertain.'),
 (760,'Testbed — Coupled, unstable vehicle dynamics','Tractor–trailers exhibit nonlinear dynamics and sensitivity to friction and terrain, with instability especially pronounced in reverse.'),
 (900,'Gap — High-speed reverse control under uncertainty','Learned predictive control remains underexplored in this demanding operating regime.')
]:
 text(title,L+12,y,31,'Bold',teal)
 para(body,L+12,y+44,cw-24,30)
pic(geom,L+40,1040,cw-80,250)
center('A demanding testbed for learning dynamics under uncertainty.',L,1310,cw,26,'Regular',gray)
bar('Experimental Setup',L,1380,cw)
labels=['(a) Asphalt','(b) Snow','(c) Slope','(d) Snow + Slope']
for i,(im,lab) in enumerate(zip(photos,labels)):
 x=L+(i%2)*546;y=1500+(i//2)*370
 text(lab,x,y,30,'Bold');pic(im,x,y+49,522,289)
para('Representative reverse-driving trajectories with learned-dynamics MPPI in BeamNG.tech.',L,2245,cw,27,gray)
para('<b>Targets:</b> −30 / −50 km/h reverse; +80 km/h forward.<br/><b>Trials:</b> 9 runs per configuration, up to 500 steps.<br/><b>Baselines:</b> Fiala-MPPI, LQR-PID, IPOPT-MPC, MPCC.',L,2334,cw,29)
# center
bar('Learning-Based MPPI',C,420,cw)
para('An MLP predicts vehicle dynamics from recent observations and controls. MPPI uses these predictions to evaluate candidate control sequences.',C+12,535,cw-24,32)
node('Observed state + control history','H = 4 steps',C+130,720,808)
arrow(C+534,834,C+534,890)
node('Learned dynamics rollout','4 hidden layers × 128 units',C+130,890,808)
arrow(C+534,1004,C+534,1060)
node('MPPI trajectory optimization','500 sampled control sequences',C+130,1060,808)
arrow(C+534,1174,C+534,1230)
node('Tractor-trailer simulation','Apply control; observe the next state',C+130,1230,808)
# feedback loop
line(C+938,1287,C+1036,1287);line(C+1036,1287,C+1036,777);arrow(C+1036,777,C+938,777)
text('Feedback',C+874,1374,25,'Regular',gray)
para('<b>Objective:</b> track target speed and the road centerline while penalizing hitch angle and constraint violations.',C+12,1430,cw-24,31)
bar('Training & Terrain Transfer',C,1590,cw)
bullets(['Collect trajectories across friction levels and target speeds; add data near handling limits.','Predict accelerations and realized actuator states, then integrate to obtain rollout states.','<b>Slope and banking are absent from training.</b>'],C+12,1705,cw-24,31)
pic(track,C+5,2050,cw-10,460)
# right
bar('Closed-Loop Results',R,420,cw)
para('Learned MPPI improves mean-speed tracking and RMS hitch angle over Fiala-MPPI in <b>all 12 environment–velocity configurations.</b>',R+12,533,cw-24,32)
text('Completed runs by environment',R+12,704,34,'Bold')
text('Three velocity setpoints pooled · maximum 27 runs per cell',R+12,754,26,'Regular',gray)
# full comparison table
cols=[R+12,R+470,R+621,R+772,R+923]
rect(R,813,cw,64,pale)
for x,s in zip(cols,['Controller','Asphalt','Snow','Slope','S + S']):text(s,x,828,27,'Bold')
rows=[('Learned MPPI',[27,27,25,19]),('Fiala-MPPI',[20,19,23,15]),('LQR-PID',[20,18,21,15]),('IPOPT-MPC',[14,11,16,13]),('MPCC',[4,4,5,4])]
for i,(lab,vals) in enumerate(rows):
 y=889+i*68
 if i==0:rect(R,y-7,cw,64,'#E8F4F2')
 text(lab,cols[0],y,29,'Bold' if i==0 else 'Regular')
 for x,v in zip(cols[1:],vals):text(str(v),x+30,y,29,'Bold' if i==0 else 'Regular')
 line(R,y+55,R+cw,y+55,'#D8E0E5',1)
text('S + S: Snow + Slope. Counts include every run.',R+12,1242,25,'Regular',gray)
para('<b>54/54 runs complete</b> on flat Asphalt and Snow. On Snow + Slope, reverse-driving stalls limit performance.',R+12,1308,cw-24,31)
bar('Multi-Step Prediction',R,1460,cw)
pic(forward,R,1576,cw,545)
para('Forward-driving rollout examples from the paper: trajectories (top) and hitch-angle error over time (bottom).',R+12,2140,cw-24,26,gray)
bullets(['Lowest position RMSE in <b>4/4 reverse</b> environments.','Lowest hitch-angle RMSE in <b>4/4 forward</b> environments.','Lowest reported RMSE in <b>13/16 comparisons</b> overall.'],R+12,2236,cw-24,30)
# small baseline footer
rect(72,2524,3312,3,'#AEBBC5')
text('* Equal contribution  |  Simulation study; hardware and payload validation remain future work.',72,2545,25,'Regular',gray)
text('48 × 36 in · Research poster mockup',2810,2545,25,'Regular',gray)
c.showPage();c.save()
