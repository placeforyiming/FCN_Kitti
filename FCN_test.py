from FCN import FCN

A=FCN()
A.build(type='FCN_8')
A.inference_object(epoch=50,im_show=0,istrain=0,refine=True)