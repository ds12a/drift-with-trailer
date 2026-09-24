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
c=canvas.Canvas(str(root/'output/pdf/tractor-trailer-poster-mockup.pdf'),pagesize=(W,H))
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
geom=crop(2,(561,86,971,304),'geometry')
asphalt=crop(6,(100,570,313,727),'asphalt')
snow=crop(6,(772,570,982,727),'snow-slope')
track=crop(6,(130,90,1007,469),'track')
rect(0,0,W,H,'#FFFFFF');rect(0,0,W,420,navy);rect(72,66,110,12,gold)
text('Learning Dynamics for High-Speed',72,110,89,'Bold','#FFFFFF')
text('Tractor-Trailer Control',72,208,89,'Bold','#FFFFFF')
text('Aaron Chen*  ·  David Shen*  ·  Taekyung Kim*  ·  Kaleb Ben Naveed  ·  Dimitra Panagou',76,325,34,'Regular','#FFFFFF')
text('UNIVERSITY OF',2770,119,31,'Bold','#FFFFFF');text('MICHIGAN',2770,159,54,'Bold',gold)
text('48 × 36 IN  /  POSTER MOCKUP',2770,332,23,'Regular','#D5E0E8')
L,C,R=72,1008,2376;lw,cw,rw=864,1296,1008
# Left column
heading('01','Why learn the dynamics?',L,478,lw)
pic(geom,L+20,565,lw-40,400)
text('Articulation makes reverse motion difficult.',L,970,27,'Bold',gray)
bullets(['Reverse motion can amplify hitch-angle errors and lead to jackknifing.','Friction, suspension, and other hidden dynamics affect high-speed behavior.','Model mismatch degrades the trajectories predicted by MPC.'],L,1040,lw,32)
heading('02','History → prediction → control',L,1430,lw)
for y,title,sub in [(1520,'Recent observations + controls','Four-step state-action history'),(1665,'Learned dynamics model','MLP predicts accelerations and actuator states'),(1810,'MPPI controller','Sample trajectories; select the next control')]:
 rect(L,y,lw,112,pale);text(title,L+24,y+13,33,'Bold');text(sub,L+24,y+61,26,'Regular',gray)
 if y<1810:text('↓',L+lw/2-10,y+115,29,'Bold',teal)
para('<b>Training:</b> varied friction and target speeds, with additional data near handling limits.',L,1975,lw,31)
para('<b>Analytic baseline:</b> reduced-order tractor-trailer dynamics with Fiala tire forces.',L,2090,lw,29)
# Center
heading('03','High-speed driving in simulation',C,478,cw)
text('REVERSE / ASPHALT',C,570,29,'Bold',teal)
text('REVERSE / SNOW + SLOPE',C+664,570,29,'Bold',teal)
pic(asphalt,C,625,632,525);pic(snow,C+664,625,632,525)
para('Representative trajectories with learned-dynamics MPPI. Colored paths show vehicle motion over time.',C,1170,cw,29,color=gray)
heading('04','Four environments. Unseen terrain.',C,1290,cw)
pic(track,C,1380,cw,585)
text('ASPHALT   /   SNOW   /   SLOPE   /   SNOW + SLOPE',C,1980,29,'Bold')
para('<b>Target velocities:</b> −30 and −50 km/h reverse; +80 km/h forward.<br/><b>Protocol:</b> nine runs per configuration in BeamNG.tech.<br/><b>Comparisons:</b> analytic-prior MPPI, LQR-PID, nonlinear MPC, MPCC.',C,2040,cw,29)
# Right
heading('05','More reliable closed-loop control',R,478,rw)
text('12/12',R,572,106,'Bold',teal)
para('configurations with closer mean-speed tracking and lower RMS hitch angle than analytic-prior MPPI.',R+328,580,rw-328,31)
text('Runs reaching the 500-step limit',R,746,34,'Bold')
text('All three velocity setpoints pooled · 27 runs per environment',R,794,25,'Regular',gray)
rect(R,855,26,14,teal);text('Learned MPPI',R+40,842,25)
rect(R+365,855,26,14,'#9BAEBE');text('Analytic-prior MPPI',R+405,842,25)
for i,(label,a,b) in enumerate([('Asphalt',27,20),('Snow',27,19),('Slope',25,23),('Snow + Slope',19,15)]):
 y=911+i*105;text(label,R,y+12,29,'Bold');bx=R+250;span=610
 rect(bx,y,span,28,pale);rect(bx,y,span*a/27,28,teal);text(f'{a}/27',bx+span+18,y-3,27)
 rect(bx,y+37,span,28,pale);rect(bx,y+37,span*b/27,28,'#9BAEBE');text(f'{b}/27',bx+span+18,y+34,27)
para('<b>54/54 runs completed</b> on flat Asphalt and Snow. Combined snow and slope still cause reverse-driving stalls.',R,1360,rw,31)
heading('06','Better multi-step prediction',R,1535,rw)
text('13/16',R,1620,106,'Bold',teal)
para('comparisons with the lowest reported open-loop RMSE.',R+328,1641,rw-328,32)
bullets(['Lowest position RMSE in all four reverse-driving environments.','Lowest hitch-angle RMSE in all four forward-driving environments.','Some analytic-prior predictions remain more accurate; gains are not universal.'],R,1780,rw,31)
para('<b>Scope:</b> simulation only. Slope and banking were absent from training; payload variation and hardware tests remain future work.',R,2095,rw,28)
# Footer
rect(0,2270,W,322,navy);rect(72,2312,12,205,gold)
text('Learned dynamics improve control under changing road conditions.',112,2310,48,'Bold','#FFFFFF')
para('Transfer to tested unseen sloped terrain, with reverse-driving limits under combined snow and slope.',112,2380,2770,34,color='#FFFFFF')
text('* Equal contribution  ·  Figures and numerical results from the manuscript  ·  Layout mockup',112,2510,24,'Regular','#D5E0E8')
rect(3020,2310,200,200,'#FFFFFF');text('VIDEO + CODE',3036,2370,22,'Bold');text('QR placeholder',3034,2407,20,'Regular',gray)
c.showPage();c.save()
