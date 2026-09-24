from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from PIL import Image
root=Path('/home/dshen/Code/drift-with-trailer')
out=root/'output/pdf/iros2026-ppniv-poster-mockup.pdf'
out.parent.mkdir(parents=True,exist_ok=True)
for name,file in [('Sans','LiberationSans-Regular.ttf'),('Bold','LiberationSans-Bold.ttf')]:
 pdfmetrics.registerFont(TTFont(name,'/usr/share/fonts/liberation-sans-fonts/'+file))
W,H=3456,2592;c=canvas.Canvas(str(out),pagesize=(W,H))
c.setTitle('Learning Dynamics for High-Speed Tractor-Trailer Control | IROS 2026 PPNIV poster')
navy='#00274C';gold='#FFCB05';ink='#173142';muted='#516574';pale='#EFF4F6';light='#D9E4E9';teal='#14758A';white='#FFFFFF'
def fill(color):c.setFillColor(HexColor(color))
def rect(x,y,w,h,color):fill(color);c.rect(x,H-y-h,w,h,fill=1,stroke=0)
def txt(s,x,y,size=29,bold=False,color=ink):
 fill(color);c.setFont('Bold' if bold else 'Sans',size);c.drawString(x,H-y-size,s)
def right(s,x,y,size=29,bold=False,color=ink):
 fill(color);c.setFont('Bold' if bold else 'Sans',size);c.drawRightString(x,H-y-size,s)
def para(s,x,y,w,size=30,color=ink,bold=False,leading=1.25):
 sty=ParagraphStyle('item',fontName='Bold' if bold else 'Sans',fontSize=size,leading=size*leading,textColor=HexColor(color),spaceAfter=0)
 P=Paragraph(s,sty);_,h=P.wrap(w,1500);P.drawOn(c,x,H-y-h);return h
def rule(x,y,w,color=light,h=3):rect(x,y,w,h,color)
def section(s,x,y,w):
 rect(x,y+8,10,47,gold);txt(s,x+25,y,40,True,navy);rule(x,y+65,w,light,3)
def image(path,x,y,w,h,cover=False):
 im=Image.open(path);iw,ih=im.size
 scale=max(w/iw,h/ih) if cover else min(w/iw,h/ih)
 dw,dh=iw*scale,ih*scale
 if cover:
  c.saveState();p=c.beginPath();p.rect(x,H-y-h,w,h);c.clipPath(p,stroke=0)
 c.drawImage(str(path),x+(w-dw)/2,H-y-(h+dh)/2,dw,dh,mask='auto')
 if cover:c.restoreState()
def bullet(s,x,y,w,size=29):
 rect(x,y+14,9,9,teal);return para(s,x+27,y,w-27,size)+22
L,C,R=72,1194,2316;cw=1068
# Page and identity
rect(0,0,W,H,white);rect(0,0,W,14,navy)
txt('IROS 2026  |  16TH PPNIV WORKSHOP',74,25,22,True,teal)
txt('Learning Dynamics for High-Speed',72,55,83,True,navy)
txt('Tractor-Trailer Control',72,147,83,True,navy)
para('Aaron Chen<super>1,*</super>   ·   David Shen<super>1,*</super>   ·   Taekyung Kim<super>2,*</super>   ·   Kaleb Ben Naveed<super>2</super>   ·   Dimitra Panagou<super>2,3</super>',76,273,2780,35,ink)
txt('1 Electrical Engineering and Computer Science   ·   2 Robotics   ·   3 Aerospace Engineering',76,317,26,False,muted)
txt('University of Michigan  ·  Ann Arbor, Michigan, USA',76,350,24,False,muted)
txt('MICHIGAN',2910,82,54,True,navy);txt('ROBOTICS',2913,144,31,True,navy)
rect(72,383,3312,132,navy)
txt('Learned dynamics improve predictive control under changing road conditions.',103,409,43,True,white)
txt('High-speed tractor-trailer driving provides a demanding testbed.',103,465,30,False,'#E6EFF3')
# Left: challenge and method
section('Why this problem?',L,584,cw)
y=690
for s in [
 '<b>Model mismatch:</b> MPC decisions depend on the accuracy of predicted vehicle motion.',
 '<b>Uncertainty:</b> friction, terrain, and hidden vehicle states change the dynamics.',
 '<b>Testbed:</b> coupled tractor-trailer motion is nonlinear and unstable in reverse.'
]: y+=bullet(s,L+6,y,cw-12,29)
image(root/'tmp/pdfs/geometry.png',L+78,1060,910,380)
para('Rigid hitch coupling makes articulation error central to reverse control.',L+40,1441,cw-80,26,muted)
section('Learned model inside MPPI',L,1510,cw)
steps=[('1  Recent observations','Four-step history of state and controls'),
       ('2  Neural dynamics','Predict accelerations and actuator states'),
       ('3  MPPI rollouts','Score 500 sampled control sequences'),
       ('4  Apply control','Observe the plant and repeat')]
for i,(name,desc) in enumerate(steps):
 yy=1610+i*151
 rect(L+18,yy,cw-36,112,pale)
 rect(L+18,yy,10,112,teal)
 txt(name,L+46,yy+12,31,True,navy)
 txt(desc,L+46,yy+60,27,False,muted)
 if i<3:txt('↓',L+cw/2-10,yy+111,31,True,teal)
para('<b>Training:</b> varied friction and speed. Slope and banking were withheld for evaluation.',L+18,2232,cw-36,28)
# Center: testbed, two visual anchors and one map
section('A demanding testbed',C,584,cw)
para('Four environments isolate changing friction and terrain; each configuration has nine trials.',C+6,687,cw-12,29)
txt('ASPHALT  ·  BASELINE',C+6,808,27,True,teal)
txt('SNOW + SLOPE  ·  COMBINED UNCERTAINTY',C+550,808,27,True,teal)
image(root/'tmp/pdfs/paper-img-005.jpg',C+6,858,510,460,True)
image(root/'tmp/pdfs/paper-img-003.jpg',C+550,858,510,460,True)
para('Representative reverse-driving trajectories with the learned controller.',C+6,1330,cw-12,26,muted)
section('What changes across the track?',C,1460,cw)
image(root/'tmp/pdfs/track.png',C+4,1561,cw-8,473)
para('<b>Friction varies in training.</b> Sloped and banked terrain appear only in evaluation.',C+6,2075,cw-12,29)
rect(C+5,2175,cw-10,149,pale)
txt('−30 / −50 km/h',C+30,2195,35,True,navy)
txt('REVERSE',C+35,2243,24,True,muted)
txt('+80 km/h',C+575,2195,35,True,navy)
txt('FORWARD',C+580,2243,24,True,muted)
# Right: result first, then predictive model evidence
section('Closed-loop control',R,584,cw)
para('Completed 500-step runs across four environments and three speed targets.',R+6,686,cw-12,29)
image(root/'tmp/pdfs/completion-poster.png',R-3,760,cw+6,659)
# Metric strips under the figure
rect(R+4,1444,506,171,pale);rect(R+531,1444,532,171,pale)
txt('98 / 108',R+30,1455,62,True,navy)
txt('completed with learned MPPI',R+30,1535,26,False,ink)
txt('62 / 72',R+557,1455,62,True,navy)
txt('reverse runs completed',R+557,1535,26,False,ink)
para('Fiala-prior MPPI completed <b>77/108</b> overall and <b>41/72</b> in reverse.',R+8,1641,cw-16,27)
section('Prediction accuracy',R,1738,cw)
txt('13 / 16',R+8,1841,89,True,teal)
para('environment–direction–metric comparisons have the lowest RMSE with learned dynamics.',R+430,1845,cw-438,29)
rule(R+8,1996,cw-16,light,3)
y=2022
for s in [
 'Lowest reverse position RMSE in <b>all four environments</b>.',
 'Lowest forward hitch-angle RMSE in <b>all four environments</b>.',
 'Combined snow and slope still causes reverse-driving stalls.'
]:y+=bullet(s,R+10,y,cw-20,28)
# Closing strip
rect(0,2392,W,200,navy)
rect(72,2424,11,121,gold)
txt('Takeaway',108,2412,38,True,gold)
para('Short-history learned dynamics improve MPPI control across tested conditions; the combined snow-and-slope case remains the hardest.',108,2463,2990,37,white)
right('* Equal contribution',3382,2551,21,False,'#DCE7EC')
c.showPage();c.save();print(out)
