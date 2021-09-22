Log_Name='ResNet50_CropNet_withoutAFN_transferToTargetDomain_RAFtoCK+'
Resume_Model='ResNet50_CropNet_withoutAFN_trainOnSourceDomain_RAFtoCK+.pkl'
##Resume_Model=None
OutputPath='.'
GPU_ID=1
Backbone='ResNet50'
useAFN='False'
methodOfAFN='SAFN'
radius=25
deltaRadius=1
weight_L2norm=0.05
useDAN='True'
methodOfDAN='DANN'
faceScale=112
sourceDataset='RAF'
targetDataset='CK+'
train_batch_size=32
test_batch_size=32
useMultiDatasets='False'
epochs=100
lr=0.0001
lr_ad=0.001
momentum=0.9
weight_decay=0.0001
isTest='False'
showFeature='False'
class_num=7
useIntraGCN='True'
useInterGCN='True'
useLocalFeature='True'
useRandomMatrix='False'
useAllOneMatrix='False'
useCov='False'
useCluster='False'
    
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=${GPU_ID} python3 TransferToTargetDomain.py \
--Log_Name ${Log_Name} \
--OutputPath ${OutputPath} \
--Backbone ${Backbone} \
--Resume_Model ${Resume_Model} \
--GPU_ID ${GPU_ID} \
--useAFN ${useAFN} \
--methodOfAFN ${methodOfAFN} \
--radius ${radius} \
--deltaRadius ${deltaRadius} \
--weight_L2norm ${weight_L2norm} \
--useDAN ${useDAN} \
--methodOfDAN ${methodOfDAN} \
--faceScale ${faceScale} \
--sourceDataset ${sourceDataset} \
--targetDataset ${targetDataset} \
--train_batch_size ${train_batch_size} \
--test_batch_size ${test_batch_size} \
--useMultiDatasets ${useMultiDatasets} \
--epochs ${epochs} \
--lr ${lr} \
--lr_ad ${lr_ad} \
--momentum ${momentum} \
--weight_decay ${weight_decay} \
--isTest ${isTest} \
--showFeature ${showFeature} \
--class_num ${class_num} \
--useIntraGCN ${useIntraGCN} \
--useInterGCN ${useInterGCN} \
--useLocalFeature ${useLocalFeature} \
--useRandomMatrix ${useRandomMatrix} \
--useAllOneMatrix ${useAllOneMatrix} \
--useCov ${useCov} \
--useCluster ${useCluster}
