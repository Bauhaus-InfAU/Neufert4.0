import gradio

import torch.nn as nn
import torch
from torch_geometric.loader import DataLoader

import utils.clean_data as cd
import utils.shape_features as sf
import utils.node_features as nf
import utils.edge_features as ef

# from datetime import datetime
# start_time = datetime.now()


node_model_path = 'utils/emb_model/Node_64.pt'
edge_model_path = 'utils/emb_model/Edge_64.pt'


class InfoGraph(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super(InfoGraph, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = False

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

        self.local_d = FF(self.embedding_dim)
        self.global_d = FF(self.embedding_dim)
        # self.local_d = MI1x1ConvNet(self.embedding_dim, mi_units)
        # self.global_d = MIFCNet(self.embedding_dim, mi_units)

        if self.prior:
            self.prior_d = PriorDiscriminator(self.embedding_dim)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, num_graphs):
        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y, M = self.encoder(x, edge_index, batch)

        g_enc = self.global_d(y)
        l_enc = self.local_d(M)

        mode='fd'
        measure='JSD'
        local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)

        if self.prior:
            prior = torch.rand_like(y)
            term_a = torch.log(self.prior_d(prior)).mean()
            term_b = torch.log(1.0 - self.prior_d(y)).mean()
            PRIOR = - (term_a + term_b) * self.gamma
        else:
            PRIOR = 0

        return local_global_loss + PRIOR


def outline_embedding(wkt, wall):
    wall_f, wkt_f = cd.read_wall_wkt(wall, wkt)
    apa_wall, apa_geo = cd.clean_geometry(wall_f, wkt_f)

    apa_geo = apa_geo
    apa_line = apa_geo.boundary

    apa_wall_O = cd.exterior_wall(apa_line, apa_wall)
    apa_coor = cd.geo_coor(apa_geo)

    xarr4cv, yarr4cv = apa_geo.exterior.coords.xy
    x4cv = xarr4cv.tolist()
    y4cv = yarr4cv.tolist()

    scale = 100000
    xmin_abs = abs(min(x4cv))
    ymin_abs = abs(min(y4cv))

    p_4_cv = cd.points4cv(x4cv, y4cv, xmin_abs, ymin_abs, scale)

    grid_points = cd.gridpoints(apa_geo, 1)

    Dir_S_longestedge, Dir_N_longestedge, Dir_W_longestedge, Dir_E_longestedge, Dir_S_max, Dir_N_max, Dir_W_max, Dir_E_max, Facade_length, Facade_ratio = sf.wall_direction_ratio(apa_line, apa_wall)
    Perimeter = sf.apartment_perimeter(apa_geo)
    Area = sf.apartment_area(apa_geo)
    BBox_width_x, BBox_height_y, Aspect_ratio, Extent, ULC_x, ULC_y, LRC_x, LRC_y = sf.boundingbox_features(apa_geo)
    Max_diameter = sf.max_diameter(apa_geo)
    Fractality = sf.fractality(apa_geo)
    Circularity = sf.circularity(apa_geo)
    Outer_radius = sf.outer_radius(p_4_cv, xmin_abs, ymin_abs, scale)
    Inner_radius = sf.inner_radius(apa_geo, apa_line)
    Dist_mean, Dist_sigma, Roundness = sf.roundness_features(apa_line)
    Compactness = sf.compactness(apa_geo)
    Equivalent_diameter = sf.equivalent_diameter(apa_geo)
    Shape_membership_index = sf.shape_membership_index(apa_line)
    Convexity, Hull_geo = sf.convexity(p_4_cv, apa_geo, xmin_abs, ymin_abs, scale)
    Rectangularity, Rect_phi, Rect_width, Rect_height = sf.rectangle_features(p_4_cv, apa_geo, xmin_abs, ymin_abs, scale)
    Squareness = sf.squareness(apa_geo)
    Moment_index = sf.moment_index(apa_geo, Convexity, Compactness)
    nDetour_index = sf.ndetour_index(apa_geo, Hull_geo)
    nCohesion_index = sf.ncohesion_index(apa_geo, grid_points)
    nProximity_index, nSpin_index = sf.nproximity_nspin_index(apa_geo, grid_points)
    nExchange_index = sf.nexchange_index(apa_geo)
    nPerimeter_index = sf.nperimeter_index(apa_geo)
    nDepth_index = sf.ndepth_index(apa_geo, apa_line, grid_points)
    nGirth_index = sf.ngirth_index(apa_geo, Inner_radius)
    nRange_index = sf.nrange_index(apa_geo, Outer_radius)
    nTraversal_index = sf.ntraversal_index(apa_geo, apa_line)

    shape = [Dir_S_longestedge, Dir_N_longestedge, Dir_W_longestedge, Dir_E_longestedge, Dir_S_max, Dir_N_max, Dir_W_max, Dir_E_max, Facade_length, Facade_ratio,
	         Perimeter, Area,
	         BBox_width_x, BBox_height_y, Aspect_ratio, Extent, ULC_x, ULC_y, LRC_x, LRC_y,
	         Max_diameter, Fractality, Circularity, Outer_radius, Inner_radius,
	         Dist_mean, Dist_sigma, Roundness,
	         Compactness, Equivalent_diameter, Shape_membership_index, Convexity,
	         Rectangularity, Rect_phi, Rect_width, Rect_height,
	         Squareness, Moment_index, nDetour_index, nCohesion_index,
	         nProximity_index, nExchange_index, nSpin_index, nPerimeter_index,
	         nDepth_index, nGirth_index, nRange_index, nTraversal_index]
    shape = [float(i) for i in shape]
    

    node_graph = nf.node_graph(apa_coor, apa_geo)
    node_model = torch.load(node_model_path)
    node_model.eval()

    node_dataloader = DataLoader(node_graph, batch_size=1)
    node_emb = node_model.encoder.get_embeddings(node_dataloader)
    node = node_emb[0].tolist()


    edge_graph = ef.edge_graph(apa_line, apa_wall)
    edge_model = torch.load(edge_model_path)
    edge_model.eval()
    
    edge_dataloader = DataLoader(edge_graph, batch_size=1)   
    edge_emb = edge_model.encoder.get_embeddings(edge_dataloader)
    edge = edge_emb[0].tolist()

    json = {"edge": edge,
            "shape": shape,
            "node": node}


    return json




gradio_interface = gradio.Interface(fn=outline_embedding,
                                    inputs = [gradio.Textbox(type="text", label="wkt", placeholder="wkt"),
                                              gradio.Textbox(type="text", label="wall", placeholder="wall")],
                                    outputs = "json",
                                    title="outline embedding")


# end_time = datetime.now()
# print('Duration: {}'.format(end_time - start_time))
# api_open=True, 
gradio_interface.queue(max_size=5, status_update_rate="auto")
gradio_interface.launch(show_error=True, enable_queue=True)

