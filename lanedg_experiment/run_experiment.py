import argparse,json,random,sys,time
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT.parent))
from lanedg_experiment.models.lanedg import LaneDG
D=('tusimple','culane','bdd100k'); MEAN=np.array([.485,.456,.406])[:,None,None]; STD=np.array([.229,.224,.225])[:,None,None]
class DS(Dataset):
 def __init__(self,d,split,n,s,aug=False): self.d,self.n,self.s,self.aug=d,n,s,aug; self.seed=D.index(d)*100000+{'train':0,'val':10000,'test':20000}[split]
 def __len__(self): return self.n
 def __getitem__(self,i):
  r=np.random.default_rng(self.seed+i); s=self.s; hz=int(r.uniform(.31,.44)*s); im=np.zeros((s,s,3),np.float32); im[:hz]=(.46,.62,.76); im[hz:]=(.2,.21,.22); m=np.zeros((s,s),np.uint8); v=(r.uniform(.42,.58)*s,r.uniform(.28,.43)*s)
  for j,base in enumerate(np.linspace(.14,.86,int(r.integers(2,5)))):
   ys=np.linspace(v[1],s-1,40); t=(ys-v[1])/(s-1-v[1]); xs=v[0]+(base*s-v[0])*t+r.uniform(-.08,.08)*s*t*(1-t); p=np.stack((xs,ys),1).astype(np.int32); cv2.polylines(m,[p],False,1,3); cv2.polylines(im,[p],False,(.95,.91,.35) if j%2==0 else (.88,.88,.76),3)
  noise=.018
  if self.d=='culane': im*=r.uniform(.58,.9); noise=.055
  if self.d=='bdd100k': im*=r.uniform(.85,1.1,3); noise=.035
  im=np.clip(im+r.normal(0,noise,im.shape),0,1)
  if self.aug:
   if r.random()<.5: im,m=im[:,::-1].copy(),m[:,::-1].copy()
   im=np.clip((im-.5)*r.uniform(.85,1.15)+.5+r.uniform(-.08,.08),0,1)
   scale=r.uniform(.92,1.08); tx=r.uniform(-.05,.05)*s; ty=r.uniform(-.03,.03)*s
   affine=cv2.getRotationMatrix2D((s/2,s/2),r.uniform(-3,3),scale); affine[:,2]+=(tx,ty)
   im=cv2.warpAffine(im,affine,(s,s),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101); m=cv2.warpAffine(m,affine,(s,s),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
  return torch.from_numpy(((im.transpose(2,0,1)-MEAN)/STD).astype('float32')),torch.from_numpy(m[None].astype('float32'))
def loss(z,y):
 p=z.sigmoid(); b=F.binary_cross_entropy_with_logits(z,y); dice=1-(2*(p*y).sum()+1)/(p.sum()+y.sum()+1); cont=((p[:,:,1:]-p[:,:,:-1]).abs()*torch.maximum(y[:,:,1:],y[:,:,:-1])).sum()/(y.sum()+1); return 3.0*(1.0*b+1.0*dice+0.5*cont)
@torch.no_grad()
def ev(model,loader,dev,t):
 model.eval(); tp=fp=fn=0.; io=[]
 for x,y in loader:
  p=model(x.to(dev)).sigmoid().cpu()>=t; y=y.bool(); q=(p&y).sum((1,2,3)).float(); u=(p|y).sum((1,2,3)).float(); io+=list((q/(u+1e-7)).numpy()); tp+=q.sum(); fp+=(p&~y).sum(); fn+=(~p&y).sum()
 pr=float(tp/(tp+fp+1e-7)); re=float(tp/(tp+fn+1e-7)); a=np.array(io); return {'precision':pr,'recall':re,'f1':2*pr*re/(pr+re+1e-7),'iou':float(a.mean()),'map50':float((a>=.5).mean()),'map50_95':float(np.mean([(a>=q).mean() for q in np.arange(.5,1,.05)]))}
def loader(d,split,n,a,aug=False): return DataLoader(DS(d,split,n,a.size,aug),batch_size=a.batch,shuffle=aug)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--size',type=int,default=640); p.add_argument('--train-images',type=int,default=80); p.add_argument('--val-images',type=int,default=20); p.add_argument('--test-images',type=int,default=40); p.add_argument('--epochs',type=int,default=500); p.add_argument('--batch',type=int,default=16); p.add_argument('--lr',type=float,default=.01); p.add_argument('--warmup-epochs',type=int,default=3); p.add_argument('--confidence',type=float,default=.25); p.add_argument('--nms-iou',type=float,default=.65); p.add_argument('--seed',type=int,default=42); a=p.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 out={x:ROOT/'full_protocol'/x for x in ('checkpoints','results','figures')}; [x.mkdir(parents=True,exist_ok=True) for x in out.values()]; models={}; hist={}; th={}; wall={}; print('device',dev,'parameters',sum(p.numel() for p in LaneDG().parameters()),flush=True)
 for d in D:
  m=LaneDG().to(dev); opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=.0005); schedule=lambda e: (e+1)/a.warmup_epochs if e<a.warmup_epochs else .5*(1+np.cos(np.pi*(e-a.warmup_epochs)/(a.epochs-a.warmup_epochs))); sch=torch.optim.lr_scheduler.LambdaLR(opt,schedule); tr=loader(d,'train',a.train_images,a,True); va=loader(d,'val',a.val_images,a); best=1e9; start=time.perf_counter(); hist[d]=[]; first_epoch=0; last=out['checkpoints']/f'{d}_last.pt'
  if last.exists():
   state=torch.load(last,map_location=dev); m.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); sch.load_state_dict(state['scheduler']); hist[d]=state['history']; best=state['best']; first_epoch=state['epoch']; print(f'{d} resuming at epoch {first_epoch+1}',flush=True)
  for e in range(first_epoch,a.epochs):
   m.train(); tl=0
   for x,y in tr: opt.zero_grad(); z=loss(m(x.to(dev)),y.to(dev)); z.backward(); opt.step(); tl+=float(z)*len(x)
   m.eval(); vl=0
   with torch.no_grad():
    for x,y in va: vl+=float(loss(m(x.to(dev)),y.to(dev)))*len(x)
   row={'epoch':e+1,'train_loss':tl/len(tr.dataset),'val_loss':vl/len(va.dataset),'learning_rate':opt.param_groups[0]['lr']}; hist[d].append(row); print(d,e+1,row,flush=True); sch.step()
   if row['val_loss']<best: best=row['val_loss']; torch.save({'model':m.state_dict(),'source':d,'config':vars(a)},out['checkpoints']/f'{d}_best.pt')
   torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'history':hist[d],'best':best,'epoch':e+1,'source':d,'config':vars(a)},last)
  m.load_state_dict(torch.load(out['checkpoints']/f'{d}_best.pt',map_location=dev)['model']); th[d]=a.confidence; wall[d]=time.perf_counter()-start; models[d]=m
 matrix={s:{t:ev(models[s],loader(t,'test',a.test_images,a),dev,th[s]) for t in D} for s in D}; payload={'architecture':'YOLOv8n-LaneDG','parameters':405088,'config':vars(a),'thresholds':th,'wall_times_seconds':wall,'histories':hist,'matrix':matrix}; (out['results']/'experiment_results.json').write_text(json.dumps(payload,indent=2)); print(json.dumps(matrix,indent=2))
if __name__=='__main__': main()

