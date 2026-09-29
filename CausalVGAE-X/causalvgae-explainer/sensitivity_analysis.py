   
                                                            
                                                       
                                                                                 
                                                                                                 
                                                                                              


       
                                               
                                               
   

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from torch_geometric.nn import GCNConv
from tqdm import tqdm
import warnings
import argparse
warnings.filterwarnings('ignore')

                                                                              
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                                                                              
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH   = os.path.join(SCRIPT_DIR, "best_causalvgae_explainer.pth")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "results", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

                                                                              
GAMMA_VALUES  = [0.5, 0.7, 0.9]
MMAX_VALUES   = [32, 64, 128]
SPARSITY_LEVELS = np.linspace(0.05, 1.0, 20)                         

                                                                            
                                          
                                                                               
                                 
                                                                      
ABLATION_VARIANTS = {
    'Complet':       {'w_recon': 1.0, 'w_kl': 0.1,  'w_spa': 5.0,  'w_temp': 0.1, 'structural': 'full'},
    'CausalVGAE':    {'w_recon': 1.0, 'w_kl': 0.1,  'w_spa': 5.0,  'w_temp': 0.1, 'structural': 'vgae_only'},
    'L_ind (ortho.)':{'w_recon': 1.0, 'w_kl': 0.0,  'w_spa': 5.0,  'w_temp': 0.1, 'structural': 'full'},
    'L_spa (parcim.)':{'w_recon': 1.0, 'w_kl': 0.1, 'w_spa': 0.0, 'w_temp': 0.1, 'structural': 'full'},
    'L_kl (rÃ©gul.)': {'w_recon': 1.0, 'w_kl': 0.0,  'w_spa': 5.0,  'w_temp': 0.1, 'structural': 'full'},
}

ABLATION_FINE_TUNE_EPOCHS = 3                                            
ABLATION_LR = 1e-4                                             

                                                                              
                                                                           
                                                                              

class TimeEncoder(nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

    def forward(self, timestamps):
        if timestamps.dim() == 0:
            timestamps = timestamps.unsqueeze(0)
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(-1)
        return self.encoder(timestamps)


class CausalVGAE_Explainer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32,
                 causal_dim=16, dropout=0.1):
        super().__init__()
        self.causal_dim  = causal_dim
        self.latent_dim  = latent_dim

        self.gc1       = GCNConv(input_dim, hidden_dim)
        self.gc2       = GCNConv(hidden_dim, hidden_dim)
        self.gc_mu     = GCNConv(hidden_dim, latent_dim)
        self.gc_logvar = GCNConv(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.mask_net = nn.Sequential(
            nn.Linear(causal_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def encode(self, x, edge_index):
        h = F.relu(self.gc1(x, edge_index))
        h = self.dropout(h)
        h = F.relu(self.gc2(h, edge_index))
        h = self.dropout(h)
        mu     = self.gc_mu(h, edge_index)
        logvar = self.gc_logvar(h, edge_index)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)

        z_causal   = z[:, :self.causal_dim]
        z_spurious = z[:, self.causal_dim:]

        mask_logits = self.mask_net(z_causal)
        causal_mask = torch.sigmoid(mask_logits / 0.5)

        recon = self.decoder(z)

        return {
            'z': z, 'z_causal': z_causal, 'z_spurious': z_spurious,
            'mu': mu, 'logvar': logvar, 'recon': recon,
            'causal_mask': causal_mask, 'mask_logits': mask_logits
        }


class CausalVGAEExplainer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32,
                 causal_dim=16, time_dim=16, dropout=0.1):
        super().__init__()
        self.time_encoder = TimeEncoder(time_dim)
        self.vgae = CausalVGAE_Explainer(input_dim + time_dim,
                               hidden_dim, latent_dim, causal_dim, dropout)
        self.target_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, snapshot, prev_z_causal=None):
        x          = snapshot['node_features']
        edge_index = snapshot['edge_index']
        timestamps = snapshot['timestamps']

        time_emb = self.time_encoder(timestamps)
        if time_emb.dim() == 2:
            time_emb = time_emb.squeeze(0)

        num_nodes = x.size(0)
        time_emb_expanded = time_emb.expand(num_nodes, -1)
        x_temporal = torch.cat([x, time_emb_expanded], dim=-1)

        vgae_output = self.vgae(x_temporal, edge_index)
        predictions = self.target_predictor(vgae_output['z'])

        temporal_loss = torch.tensor(0.0, device=x.device)
        if prev_z_causal is not None:
            ms = min(vgae_output['z_causal'].size(0), prev_z_causal.size(0))
            temporal_loss = F.mse_loss(
                vgae_output['z_causal'][:ms], prev_z_causal[:ms])

        return {**vgae_output, 'predictions': predictions,
                'temporal_loss': temporal_loss}

                                                                              
                                                           
                                                                              

class TemporalDataProcessor:
    def __init__(self, csv_path, window_size=100, overlap=50):
        self.csv_path    = csv_path
        self.window_size = window_size
        self.overlap     = overlap

    def load_and_preprocess(self):
        print(f"Loading {os.path.basename(self.csv_path).replace('.csv', '').title()} dataset ...")
        df = pd.read_csv(self.csv_path)
        print(f"  shape: {df.shape}")
        ts = df['timestamp'].values.astype(float)
        ts = (ts - ts.min()) / (ts.max() - ts.min() + 1e-8)
        features_raw = df['comma_separated_list_of_features'].values
        return {
            'user_ids':   df['user_id'].values,
            'item_ids':   df['item_id'].values,
            'timestamps': ts,
            'labels':     df['state_label'].values,
            'features':   features_raw,
            'n_features': 172
        }

    def create_temporal_snapshots(self, data):
        print("Creating temporal snapshots ...")
        ts = data['timestamps']
        n_win = max(1,
            (len(ts) - self.window_size) // (self.window_size - self.overlap) + 1)
        snapshots = []
        for i in range(n_win):
            s = i * (self.window_size - self.overlap)
            e = min(s + self.window_size, len(ts))
            wd = {k: data[k][s:e] for k in
                  ['user_ids', 'item_ids', 'timestamps', 'labels', 'features']}
            wd['window_id']  = i
            wd['time_range'] = (ts[s], ts[e-1])
            adj, nf = self._build_graph(wd)
            wd['adj_matrix']    = adj
            wd['node_features'] = nf
            snapshots.append(wd)
        print(f"  -> {len(snapshots)} snapshots")
        return snapshots

    def _build_graph(self, wd):
        uids, iids = wd['user_ids'], wd['item_ids']
        feats_str  = wd['features']
        parsed = []
        for fs in feats_str:
            if isinstance(fs, str):
                try:    fa = np.array([float(x) for x in fs.split(',')])
                except: fa = np.zeros(172)
            else:
                fa = np.zeros(172)
                fa[0] = float(fs) if not np.isnan(float(fs)) else 0.0
            parsed.append(fa)
        feats = np.array(parsed)
        if feats.shape[1] != 172:
            nf2 = np.zeros((feats.shape[0], 172))
            mc  = min(feats.shape[1], 172)
            nf2[:, :mc] = feats[:, :mc]
            feats = nf2

        nodes   = np.unique(np.concatenate([uids, iids]))
        nmap    = {n: i for i, n in enumerate(nodes)}
        nn_     = len(nodes)
        adj     = np.zeros((nn_, nn_))
        for u, it in zip(uids, iids):
            adj[nmap[u], nmap[it]] = 1
            adj[nmap[it], nmap[u]] = 1
        nf = np.zeros((nn_, 172))
        for nd, idx in nmap.items():
            mask = (uids == nd) | (iids == nd)
            if mask.any():
                nf[idx] = feats[mask].mean(axis=0)
        return adj, nf


def prepare_snapshots(snapshots):
    prepared = []
    for sn in snapshots:
        adj = sn['adj_matrix']
        ei  = []
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i, j] > 0:
                    ei.append([i, j])
        if not ei:
            ei = [[0, 0]]
        prepared.append({
            'node_features': torch.tensor(sn['node_features'],
                                          dtype=torch.float32).to(DEVICE),
            'edge_index':    torch.tensor(ei, dtype=torch.long).t().to(DEVICE),
            'labels':        torch.tensor(sn['labels'],
                                          dtype=torch.float32).to(DEVICE),
            'timestamps':    torch.tensor(float(sn['timestamps'].mean()),
                                          dtype=torch.float32).to(DEVICE),
        })
    return prepared

                                                                              
                                        
                                                                              

def compute_fidelity_at_sparsity(model, snapshots, sparsity_level,
                                  gamma=0.7, m_max=64):
       
                                                              

              
              
                                    
                                                        
                 
                                                                                
                                                                       
                                                            
               
                                                                              
                                                     
       
    model.eval()
    fid_plus_list  = []
    fid_minus_list = []
    prev_z_causal  = None

    with torch.no_grad():
        for snap in snapshots:
            output = model(snap, prev_z_causal)

            original_pred = output['predictions']        
            causal_mask   = output['causal_mask']         
            z_causal      = output['z_causal']

            N = causal_mask.size(0)
            num_edges = snap['edge_index'].size(1)

            
            t = snap['timestamps'].item()
            recency = torch.full((N, 1), t, device=DEVICE)  
            adjusted_mask = (causal_mask ** gamma) * (recency ** (1.0 - gamma))
            adjusted_mask = adjusted_mask / (adjusted_mask.max() + 1e-8)

        
            flat_mask = adjusted_mask.squeeze()
            if flat_mask.numel() > m_max:
                topk_vals, topk_idx = torch.topk(flat_mask, m_max)
                restricted_mask = torch.zeros_like(flat_mask)
                restricted_mask[topk_idx] = flat_mask[topk_idx]
            else:
                restricted_mask = flat_mask.clone()

            k = max(1, int(sparsity_level * restricted_mask.numel()))
            if k >= restricted_mask.numel():
                binary_mask = torch.ones_like(restricted_mask)
            else:
                threshold = torch.topk(restricted_mask, k).values[-1]
                binary_mask = (restricted_mask >= threshold).float()

            binary_mask_2d = binary_mask.unsqueeze(-1) 


            masked_pred  = original_pred * binary_mask_2d
            orig_class   = (original_pred > 0.5).float()
            masked_class = (masked_pred   > 0.5).float()
            min_sz       = min(orig_class.size(0), masked_class.size(0))
            match        = (orig_class[:min_sz] == masked_class[:min_sz]).float()
            delta        = torch.abs(original_pred[:min_sz] - masked_pred[:min_sz])
            proximity    = (1.0 - delta).mean()
            fid_p        = 0.7 * match.mean() + 0.3 * proximity
            fid_plus_list.append(fid_p.item())

            
            inv_mask        = 1.0 - binary_mask_2d
            non_causal_pred = original_pred * inv_mask
            nc_class        = (non_causal_pred > 0.5).float()
            change          = (orig_class[:min_sz] != nc_class[:min_sz]).float()
            delta_nc        = torch.abs(original_pred[:min_sz]
                                        - non_causal_pred[:min_sz])
            fid_m           = 0.6 * change.mean() + 0.4 * delta_nc.mean()
            fid_minus_list.append(fid_m.item())

            prev_z_causal = z_causal.detach()

    return float(np.mean(fid_plus_list)), float(np.mean(fid_minus_list))


def run_sensitivity_sweep(model, snapshots):
    results = {'gamma': {}, 'mmax': {}}

    print("\n=== Sweeping gamma (m_max = 64) ===")
    for gamma in GAMMA_VALUES:
        fp_curve, fm_curve = [], []
        for sp in tqdm(SPARSITY_LEVELS, desc=f"gamma={gamma}"):
            fp, fm = compute_fidelity_at_sparsity(
                model, snapshots, sp, gamma=gamma, m_max=64)
            fp_curve.append(fp)
            fm_curve.append(fm)
        results['gamma'][gamma] = {
            'fidelity_plus':  np.array(fp_curve),
            'fidelity_minus': np.array(fm_curve),
            'sparsity':       SPARSITY_LEVELS.copy()
        }

                                         
    print("\n=== Sweeping m_max (gamma = 0.7) ===")
    for mmax in MMAX_VALUES:
        fp_curve, fm_curve = [], []
        for sp in tqdm(SPARSITY_LEVELS, desc=f"m_max={mmax}"):
            fp, fm = compute_fidelity_at_sparsity(
                model, snapshots, sp, gamma=0.7, m_max=mmax)
            fp_curve.append(fp)
            fm_curve.append(fm)
        results['mmax'][mmax] = {
            'fidelity_plus':  np.array(fp_curve),
            'fidelity_minus': np.array(fm_curve),
            'sparsity':       SPARSITY_LEVELS.copy()
        }

    return results



def plot_sensitivity(results, output_dir, dataset_name):
 

    plt.rcParams.update({
        'font.family':       'serif',
        'font.size':         11,
        'axes.labelsize':    12,
        'axes.titlesize':    12,
        'legend.fontsize':   10,
        'xtick.labelsize':   10,
        'ytick.labelsize':   10,
        'lines.linewidth':   1.8,
        'axes.linewidth':    0.8,
        'grid.alpha':        0.3,
        'grid.linewidth':    0.5,
        'figure.dpi':        300,
        'savefig.dpi':       300,
        'savefig.bbox':      'tight',
        'text.usetex':       False,   
    })

    
    COLORS = {
        0.5:  '#d62728', 
        0.7:  '#1f77b4',   
        0.9:  '#2ca02c',   
        32:   '#d62728',
        64:   '#1f77b4',
        128:  '#2ca02c',
    }

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0),
                             gridspec_kw={'hspace': 0.42, 'wspace': 0.28})

                                         
    def _plot_panel(ax, data_dict, param_key, metric_key, ylabel,
                    title, legend_fmt):
        for val, data in data_dict.items():
            sparsity = data['sparsity']
            curve    = data[metric_key]
            label    = legend_fmt.format(val)
            ax.plot(sparsity, curve, color=COLORS[val], label=label,
                    linewidth=1.8, solid_capstyle='round')

        ax.set_xlim(0.0, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel('Sparsity')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold', pad=8)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.grid(True, linestyle='-', alpha=0.25)
        ax.legend(loc='lower right', framealpha=0.9,
                  edgecolor='#cccccc', fancybox=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

                            
    _plot_panel(axes[0, 0], results['gamma'], 'gamma', 'fidelity_plus',
                'Fidelity+',
                r'(a) Exploration/Exploitation ($\gamma$)',
                r'$\gamma$ = {0}')
    _plot_panel(axes[0, 1], results['mmax'], 'mmax', 'fidelity_plus',
                'Fidelity+',
                r'(b) Candidate Size ($m_{max}$)',
                r'$m_{{max}}$ = {0}')

                            
    _plot_panel(axes[1, 0], results['gamma'], 'gamma', 'fidelity_minus',
                r'Fidelity$-$',
                r'(a) Exploration/Exploitation ($\gamma$)',
                r'$\gamma$ = {0}')
    _plot_panel(axes[1, 1], results['mmax'], 'mmax', 'fidelity_minus',
                r'Fidelity$-$',
                r'(b) Candidate Size ($m_{max}$)',
                r'$m_{{max}}$ = {0}')

                                  
    fig.text(0.27, 0.97, r'Spatio-Temporal Trade-off ($\gamma$)',
             ha='center', va='top', fontsize=13, fontweight='bold')
    fig.text(0.75, 0.97, r'Candidate Search Space ($m_{max}$)',
             ha='center', va='top', fontsize=13, fontweight='bold')

    fig.text(0.5, -0.01,
             f'Fig. 5: Sensitivity analysis of causalvgae-explainer '
             f'on the {dataset_name.title()} dataset.',
             ha='center', va='top', fontsize=11, style='italic')

                
    output_stem = f'sensitivity_{dataset_name.lower()}'
    pdf_path = os.path.join(output_dir, f'{output_stem}.pdf')
    png_path = os.path.join(output_dir, f'{output_stem}.png')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    fig.savefig(png_path, format='png', bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved:  {pdf_path}")
    print(f"Saved:  {png_path}")
    return png_path


def main():
    parser = argparse.ArgumentParser(description='Run causalvgae-explainer sensitivity analysis.')
    parser.add_argument('--dataset', choices=['wikipedia', 'reddit'], default='wikipedia')
    parser.add_argument('--checkpoint', default=CKPT_PATH)
    parser.add_argument('--max-snapshots', type=int, default=None,
                        help='Limit evaluation to the last N snapshots.')
    args = parser.parse_args()

    data_path = os.path.join(SCRIPT_DIR, "..", "resources", "datasets", "raw",
                             f"{args.dataset}.csv")
    print("=" * 65)
    print("  Hyperparameter Sensitivity Analysis -- CausalVgae_explainer")
    print(f"  Dataset: {args.dataset.title()}  |  gamma in {{0.5, 0.7, 0.9}}  |  m_max in {{32, 64, 128}}")
    print("=" * 65)

                  
    processor = TemporalDataProcessor(data_path, window_size=100, overlap=50)
    data      = processor.load_and_preprocess()
    snapshots = processor.create_temporal_snapshots(data)
    prepared  = prepare_snapshots(snapshots)
    if args.max_snapshots is not None:
        prepared = prepared[-args.max_snapshots:]
    print(f"  {len(prepared)} snapshots ready.")

                                      
    model = CausalVGAEExplainer(input_dim=172, hidden_dim=64, latent_dim=32,
                           causal_dim=16, time_dim=16, dropout=0.1).to(DEVICE)
    ckpt  = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"  Model loaded from {args.checkpoint}")

                              
    results = run_sensitivity_sweep(model, prepared)

             
    png = plot_sensitivity(results, OUTPUT_DIR, args.dataset)

                         
    raw_path = os.path.join(OUTPUT_DIR, f'sensitivity_{args.dataset}.pth')
    torch.save(results, raw_path)
    print(f"Raw results saved: {raw_path}")

    print("\n" + "=" * 65)
    print("  SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()

