"""Export repaired racket models with explicit precision/finite-coordinate policy."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def bounded_pose_decode(self,kpts):
    import torch
    y=kpts.view(kpts.shape[0],*self.kpt_shape,-1)
    # Raw GRID offsets beyond +/-512 cannot represent an in-frame point at 960.
    # Saturate before *2 and *stride to avoid half overflow, WITHOUT snapping to image edges.
    xy=y[:,:,:2].clamp(-512,512)
    xy=(xy*2+(self.anchors-.5))*self.strides
    return torch.cat((xy,y[:,:,2:3].sigmoid()),2).view(kpts.shape[0],self.nk,-1)

def main():
    import torch
    import coremltools as ct
    from ultralytics import YOLO
    from ultralytics.nn.modules import Detect
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True);parser.add_argument('--fp32',action='store_true');parser.add_argument('--bounded',action='store_true');parser.add_argument('--imgsz',type=int,choices=(640,960),default=960);args=parser.parse_args()
    if Path(args.output).exists():raise FileExistsError(args.output)
    torch.set_num_threads(4)
    model=YOLO(args.checkpoint).model.float().cpu().eval().fuse()
    for p in model.parameters():p.requires_grad=False
    for module in model.modules():
        if isinstance(module,Detect):module.export=True;module.format='coreml';module.dynamic=False;module.shape=None
    if args.bounded:
        import types
        head=model.model[-1]
        if type(head).__name__ != 'Pose':raise ValueError('Bounded decoder validated only for the supplied legacy Pose head')
        head.kpts_decode=types.MethodType(bounded_pose_decode,head)
    image=torch.zeros((1,3,args.imgsz,args.imgsz))
    with torch.no_grad():model(image);model(image)
    traced=torch.jit.trace(model,image,strict=False)
    precision=ct.precision.FLOAT32 if args.fp32 else ct.precision.FLOAT16 if args.bounded else ct.transform.FP16ComputePrecision(op_selector=lambda op:op.op_type == 'conv')
    converted=ct.convert(traced,inputs=[ct.ImageType(name='image',shape=image.shape,scale=1/255,bias=[0.,0.,0.])],
                         convert_to='mlprogram',compute_precision=precision)
    converted.user_defined_metadata['coordinate_precision']=('FP16 with grid offsets bounded to [-512,512]' if args.bounded else 'FP32' if args.fp32 else 'FP32 attention/geometry; FP16 convolution')
    converted.user_defined_metadata['checkpoint']=str(Path(args.checkpoint).resolve())
    converted.user_defined_metadata['bounded_decoder']=str(args.bounded)
    converted.user_defined_metadata['input_resolution']=str(args.imgsz)
    converted.save(args.output)
    print('EXPORTED',args.output,flush=True)

if __name__=='__main__':main()
