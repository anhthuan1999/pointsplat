import utils
from utils.transform_utils import remove_outliers, MinMaxScaler
import torch #type: ignore
import torch.nn as nn #type: ignore
from collections import OrderedDict
from .pointtransformer_v3 import PointTransformerV3Model
from .spconv import SparseConvModel
import gin  #type: ignore
gin.external_configurable(torch.nn.Identity)
gin.external_configurable(torch.nn.Tanh)
gin.external_configurable(torch.nn.Sigmoid)
from typing import List
from .pt import weak_seg_repro as model_test


SF = False

FEATURE2CHANNEL = {
    'means': 3,
    'features_dc': 3,
    'features_rest': 3,
    'opacities': 1,
    'scales': 3,
    'quats': 4,
}


ALL_FEATURES = ['means','features_dc','features_rest','opacities','scales','quats']
@gin.configurable
class FeaturePredictor(nn.Module):
    def __init__(self, 
                 backbone_type,
                 sh_degree,
                 input_features,
                 input_feat_to_mlp,
                 output_features,
                 output_head_nlayer,
                 output_head_type,
                 output_head_width,
                 output_features_type, # 'dc:direct component or res:residual"
                 res_feature_activation,
                 max_scale_normalized,
                 grid_resolution,
                 resume_ckpt,
                 input_embed_to_mlp,
                 zeroinit,
                 ):
        super(FeaturePredictor, self).__init__()
        self.sh_degree = sh_degree
        sh_dim = (sh_degree+1)**2-1
        FEATURE2CHANNEL['features_rest'] = sh_dim*3
        self.input_features = input_features
        self.input_feat_to_mlp = input_feat_to_mlp
        in_channels = sum([FEATURE2CHANNEL[feature] for feature in input_features])
        self.gs_features_dim = in_channels
        self.output_features = output_features
        if max_scale_normalized<=0:
            print('Setting max_scale_normalized <0, turning off scale clamping')
        self.max_scale_normalized = max_scale_normalized
        self.backbone_type = backbone_type
        self.grid_resolution = grid_resolution
        self.resume_ckpt = resume_ckpt
        self.output_features_type = output_features_type 
        self.res_feature_activation = res_feature_activation 
        self.input_embed_to_mlp = input_embed_to_mlp

        if backbone_type == 'SP':
            self.backbone = SparseConvModel(in_channels=in_channels)
        elif backbone_type == 'PT':
            if SF == True:
                self.backbone = PointTransformerV3Model(in_channels=in_channels) #in_channels
            else:
                self.backbone = PointTransformerV3Model(in_channels=32) #in_channels 32
            #self.backbone = model_test(c=in_channels, k=96)
        else:
            raise NotImplementedError
        head_input_dim = self.backbone.output_dim
        #head_input_dim = 96
        if self.input_feat_to_mlp:
            if SF == True:
                head_input_dim += in_channels # in_channels       32
            else:
                head_input_dim += 32 # 32

        self.features_outputhead = nn.ModuleDict()
        for feature in output_features:
            if output_head_type=='mlp-relu':
                module_list = nn.ModuleList()
                for _ in range(output_head_nlayer-1):
                    module_list.extend(
                        [nn.Linear(head_input_dim if _==0 else output_head_width, output_head_width),
                        nn.ReLU()]
                    )
                outputdim_ = FEATURE2CHANNEL[feature]
                module_list.append(
                    nn.Linear(output_head_width if output_head_nlayer>1 else head_input_dim, outputdim_)
                )
                self.features_outputhead[feature] = nn.Sequential(*module_list)
            else:
                raise NotImplementedError
        if zeroinit:
            #init the last layer of each feature predictor to be zeros
            for k, module in self.features_outputhead.items():
                module[-1].weight.data.zero_()
                module[-1].bias.data.zero_()



        #----------------------

        self.fc_geom1 = nn.Linear(10, 64)
        self.fc_geom2 = nn.Linear(64, 10)

        # Fully Connected layers for Appearance
        self.fc_app1 = nn.Linear(in_channels-10, 64)
        self.fc_app2 = nn.Linear(64, in_channels-10)

        # Activation & Gating
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        # -----------------------------------
        geom_dim = 10 #10
        # app_dim = in_channels-10
        app_dim = 12
        hidden_dim = 32
        # Geometry-guided modulation (FiLM-like)

        self.reduce_app = nn.Sequential(
            nn.Linear(45, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),  # gamma + beta
        )

        self.modulator = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # app+geo
        )
        
        self.fusion_gate= nn.Sequential(
            nn.Linear(geom_dim+app_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, geom_dim+app_dim)
        )

        self.geom_encoder = nn.Sequential(
            nn.Linear(geom_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim) #geom_dim
        )

        self.app_encoder = nn.Sequential(
            nn.Linear(app_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim) #app_dim
        )

        # self.skip_weight = nn.Parameter(torch.tensor(0.5))

    
    def normalized_gs(self, batch_gs):
        scalers = []
        batch_normalized_gs = []
        for gs in batch_gs:
            normalized_gs = {}
            scaler = MinMaxScaler()
            scaler.fit(gs['means'])
            normalized_gs['means'] = scaler.transform(gs['means']) 
            normalized_gs['scales'] = gs['scales'] + torch.log(scaler.scale_)
            normalized_gs['features_dc'] = gs['features_dc']
            scalers.append(scaler)
            batch_normalized_gs.append(normalized_gs)
        return batch_normalized_gs, scalers

    def unnormalized_gs(self, batch_gs, scalers): #TODO
        batch_unnormalized_gs = []
        for gs, scaler in zip(batch_gs, scalers):
            unnormalized_gs = {}
            for key in gs:
                if key=='means': #The predicted gs may not contain means
                    unnormalized_gs['means'] = scaler.inverse_transform(gs['means'])
                elif key=='scales':
                    unnormalized_gs['scales'] = gs['scales'] - torch.log(scaler.scale_)
                else:
                    unnormalized_gs[key] = gs[key]
            batch_unnormalized_gs.append(unnormalized_gs)
        return  batch_unnormalized_gs

    # def forward(self, batch_gs):
    #     #1. Normalize
    #     batch_normalized_gs, batch_scalers = self.normalized_gs(batch_gs) #Move to dataloader part
    def forward(self, batch_normalized_gs, batch_scene_idx, 
                **kwargs):
        

        # start = time()
        device = batch_normalized_gs[0]['means'].device #It should be cuda
        input_keys = sorted(batch_normalized_gs[0])

        #2. Batchify
        offset = torch.tensor([gs['means'].shape[0] for gs in batch_normalized_gs]).cumsum(0)
        feat = []

        
        if SF == True:
            # CONCAT
            for bi, gs in enumerate(batch_normalized_gs):
                feat_list = []
                for key in self.input_features:
                    if key=='means':
                        feat_list.append(gs[key])
                    elif key == 'features_rest':
                        feat_list.append(gs[key].view(gs[key].shape[0], -1))
                    else:
                        feat_list.append(gs[key])
                feat.append(torch.cat(feat_list, dim=1)) #N, D
            
            feat = torch.cat(feat, dim=0) #Bx-N, D
        else:
            # # TRANS ALL
            normalized_features=batch_normalized_gs[0]

            # xyz_norm = normalized_features['means']
            # scales = normalized_features['scales']

            # log_volume = torch.log(scales.prod(-1, keepdim=True))
            # s = scales.sort(-1)[0]
            # anisotropy = torch.stack([s[:,2]/s[:,0], s[:,1]/s[:,0]],-1)
            
            f_rest = normalized_features['features_rest'].view(normalized_features['features_rest'].shape[0], -1)
            geometry = torch.cat([normalized_features['means'], normalized_features['scales'], normalized_features['quats']], dim=-1)
            
            #log_volume, anisotropy
            # print(normalized_features['features_dc'].shape)
            # appearance = torch.cat([normalized_features['features_dc'], f_rest, normalized_features['opacities']], dim=-1)
            f_reduce = self.reduce_app(f_rest)
            
            appearance = torch.cat([normalized_features['features_dc'], f_reduce, normalized_features['opacities']], dim=-1)

            pe = self.modulator(normalized_features['means']) 

            f_p = self.geom_encoder(geometry)
            f_a = self.app_encoder(appearance)
            w = self.softmax(f_a + pe)
            feat_encoder = f_p + pe
            #feat = feat_encoder

            
            # feat = self.softmax(f_a) * (f_p+pe)
            feat = w * feat_encoder
            
        
        if self.backbone_type in ['PT','SP']:
            model_input = {
                'coord': torch.cat([gs['means'] for gs in batch_normalized_gs], dim=0),
                'grid_size': torch.ones([3])*1.0/self.grid_resolution,
                'offset': offset.to(device),
                'feat': feat,
            }
            model_input['grid_coord'] = torch.floor(model_input['coord']*self.grid_resolution).int() #[0~1]/
        else:
            raise NotImplementedError

        y = self.backbone(model_input)
        #y = self.backbone([model_input['coord'],model_input['feat'],model_input['offset']])

        if self.backbone_type in ['PT']:
            #pass
            y = y['feat']

        hidden_features = y
        if self.input_feat_to_mlp:
            y = torch.cat([y, feat], dim=1)
    
        output = OrderedDict()
        for feature in self.output_features:
            feature_o = self.features_outputhead[feature](y)
            if self.output_features_type=='dc': #Predict the feature itself
                if feature == 'scales' and self.max_scale_normalized>0:
                    feature_o = torch.nn.functional.relu(feature_o)*-1
                    feature_o = feature_o + torch.log(torch.tensor(self.max_scale_normalized))
                if feature=='features_rest':
                    feature_o = feature_o.view(feature_o.shape[0], -1, 3)
                output[feature] = feature_o
            elif self.output_features_type=='res': #Predict the modulation and residual (mod first and res then)
                pointer = 0
                feature_o_res = feature_o[:, pointer:pointer+FEATURE2CHANNEL[feature]]
                feature_o_res = self.res_feature_activation[feature](feature_o_res)
                pointer += FEATURE2CHANNEL[feature]
                if feature == 'features_rest':
                    feature_o_res = feature_o_res.view(feature_o_res.shape[0], -1, 3)
                output[feature] = feature_o_res

        #-2. Unbatchify
        out_batch_normalized_gs = []
        if self.backbone_type in ['PT','SP']:
            left = 0
            for ii,(right, in_gs) in enumerate(zip(offset, batch_normalized_gs)):
                out_normalized_gs = {}
                for feature in self.output_features:
                    if self.output_features_type=='dc':
                        out_normalized_gs[feature] = output[feature][left:right]
                    elif self.output_features_type=='res':
                        
                        out_normalized_gs[feature] = in_gs[feature] + output[feature][left:right]

                out_batch_normalized_gs.append(out_normalized_gs)
                left = right

        for key in ALL_FEATURES:
            if self.sh_degree==0 and key=='features_rest':
                continue
            if key not in self.output_features: #If the feature is not in the output, we need to copy it
                for out_gs, in_gs in zip(out_batch_normalized_gs, batch_normalized_gs):
                    out_gs[key] = in_gs[key]

        assert len(out_batch_normalized_gs) == 1, 'Now only support batch size 1'
        return out_batch_normalized_gs



            