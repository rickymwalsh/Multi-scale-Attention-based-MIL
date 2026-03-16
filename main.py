# internal imports
from MIL.MIL_experiment import do_experiments
from MIL.roi_eval import ROI_Eval
from MIL.inference_MIL_classifier import Eval
from utils.generic_utils import seed_all

#external imports
import warnings
import os
import hashlib
import json
import torch
import wandb

warnings.filterwarnings("ignore")
import argparse
import yaml
import time
from datetime import datetime

from pathlib import Path


def _compute_exp_id(args, date_str: str) -> str:
    """Generate a hybrid date+hash experiment ID.

    The 8-char hex hash is deterministic: identical key configs always produce
    the same ID, enabling natural deduplication of re-runs.
    """
    hash_params = {
        'dataset': args.dataset,
        'label': args.label,
        'mil_type': args.mil_type,
        'pooling_type': args.pooling_type,
        'map_prob_func': args.map_prob_func,
        'multi_scale_model': args.multi_scale_model,
        'scales': sorted(args.scales) if args.scales else None,
        'type_scale_aggregator': args.type_scale_aggregator,
        'deep_supervision': args.deep_supervision,
        'type_mil_encoder': args.type_mil_encoder,
        'fcl_encoder_dim': args.fcl_encoder_dim,
        'fcl_dropout': args.fcl_dropout,
        'drop_attention_pool': args.drop_attention_pool,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'fpn_dim': args.fpn_dim,
        'nested_model': args.nested_model,
        'type_region_aggregator': args.type_region_aggregator,
        'feature_extraction': args.feature_extraction,
    }
    raw = json.dumps(hash_params, sort_keys=True)
    hash8 = hashlib.sha256(raw.encode()).hexdigest()[:8]
    return f"{date_str}_{hash8}"


def _build_wandb_tags(args) -> list:
    """Derive a list of tags from key config params for wandb filtering."""
    tags = [
        args.dataset,
        args.label,
        f"model_{args.multi_scale_model or 'single'}",
        f"pool_{args.pooling_type}",
        f"lr_{args.lr}",
        f"bs_{args.batch_size}",
        f"enc_dim_{args.fcl_encoder_dim}",
    ]
    if args.multi_scale_model:
        tags.append(f"scales_{'_'.join(str(s) for s in args.scales)}")
        if args.type_scale_aggregator:
            tags.append(f"agg_{args.type_scale_aggregator}")
        if args.deep_supervision:
            tags.append("deep_supervision")
    return tags

def config():
    parser = argparse.ArgumentParser()
    
    # Folders
    parser.add_argument('--output_dir', metavar='DIR',default='Mammo-CLIP-output/out_splits_new', help='path to output logs')
    parser.add_argument("--data_dir",default="datasets/Vindir-mammoclip",type=str, help="Path to data file")
    parser.add_argument("--clip_chk_pt_path", default=None, type=str, help="Path to Mammo-CLIP chkpt")
    parser.add_argument("--csv_file", default="grouped_df.csv", type=str, help="data csv file")
    parser.add_argument('--feat_dir', default='new_extracted_features', type=str)
    parser.add_argument("--img_dir", default="VinDir_preprocessed_mammoclip/images_png", type=str, help="Path to image file")

    parser.add_argument('--train', action='store_true', default=False, help='Training mode.')
    parser.add_argument('--skip_val', action='store_true', default=False, help='Skip validation each epoch (diagnostic use only).')
    parser.add_argument('--evaluation', action='store_true', default=False, help='Evaluation mode.')
    parser.add_argument('--eval_set', default='test', choices = ['val', 'test'], type=str, help="")
    
    # Data settings
    parser.add_argument("--img-size", nargs='+', default=[1520, 912])
    parser.add_argument("--dataset", default="ViNDr", type=str, help="Dataset name.")
    parser.add_argument("--data_frac", default=1.0, type=float, help="Fraction of data to be used for training")
    parser.add_argument("--label", default="Mass", type=str, help="Mass or Suspicious_Calcification")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--n_runs", default=1, type=int)
    parser.add_argument("--start_run", default=0, type=int)
    parser.add_argument('--val_split', type=float, default=0.2, help='val split ratio (default: 0.2)')
    parser.add_argument("--n_folds", default=1, type=int)
    parser.add_argument("--start-fold", default=0, type=int)
    parser.add_argument("--mean", default=0.3089279, type=float)
    parser.add_argument("--std", default=0.25053555408335154, type=float)

    # Mammo-CLIP settings 
    parser.add_argument('--model-type', default="Classifier", type=str)
    parser.add_argument("--arch", default="upmc_breast_clip_det_b5_period_n_ft", type=str,
        help="For b5 classification, [upmc_breast_clip_det_b5_period_n_lp for linear probe and  upmc_breast_clip_det_b5_period_n_ft for finetuning]. "
             "For b2 classification, [upmc_breast_clip_det_b2_period_n_lp for linear probe and  upmc_breast_clip_det_b2_period_n_ft for finetuning].")
    parser.add_argument("--swin_encoder", default="microsoft/swin-tiny-patch4-window7-224", type=str)
    parser.add_argument("--pretrained_swin_encoder", default="y", type=str)
    parser.add_argument("--swin_model_type", default="y", type=str)

    parser.add_argument("--feature_extraction", default = 'offline', type = str) 
    parser.add_argument("--feat_dim", default = 352, type = int) 
    parser.add_argument('--final_pooling', default='maxpool', choices = ['maxpool', 'pool4'], type=str, help="Which pooling operation to apply after the FPN feature maps. By default, the original from the paper is applied: max pooling with stride=4 and kernel_size=1. option pool4=> kernel_size=4, stride=4.")
    
    # Patch extraction 
    parser.add_argument('--patching', action = 'store_true', default = False, help = 'Wether to perform patching on full-resolution images. If false, it will consider previously extracted patches that were saved in a directory (default: False)')
    parser.add_argument('--source_image', type = str, default = 'patches', choices = ['patches', 'full_image'])
    parser.add_argument('--patch_size', type = int, default = 512) 
    parser.add_argument('--overlap', type = float, nargs='*',  default=[0.0])
    
    # MIL model parameters
    parser.add_argument('--mil_type', default=None, choices=[None, 'instance', 'embedding', 'pyramidal_mil'], type=str, help="MIL approach")
    parser.add_argument('--pooling_type', default='mean', choices=['max', 'mean', 'attention', 'gated-attention', 'pma'], type=str, help="MIL pooling operator")
    parser.add_argument('--type_mil_encoder', default='mlp', choices=['mlp', 'sab', 'isab'], type=str, help="Type of MIL encoder.")
    
    parser.add_argument('--fcl_attention_dim', type=int, default=128, metavar='N', help='parameter for attention (internal hidden units)')
    parser.add_argument('--map_prob_func', type=str, default = 'softmax', choices = ['softmax', 'sparsemax', 'entmax', 'alpha_entmax'])

    parser.add_argument('--fcl_encoder_dim', type=int, default=256, help='parameter for set transformer (internal hidden units)')
    parser.add_argument('--sab_num_heads', type=int, default=4, help='parameter for set transformer (number of self-attention heads in set attention blocks)')
    parser.add_argument('--isab_num_heads', type=int, default=4, help='parameter for set transformer (number of self-attention heads in induced set attention blocks)')
    parser.add_argument('--pma_num_heads', type=int, default=1, help='parameter for set transformer (number of self-attention heads in pooling by multihead attention)')
    parser.add_argument('--num_encoder_blocks', type=int, default=2, help='parameter for set transformer (number of encoder layers)')
    parser.add_argument('--trans_num_inds', type=int, default=20, help='parameter for set transformer (number of inducing points for the ISAB)')
    parser.add_argument('--trans_layer_norm', type=bool, default=False)

    #Multi-scale MIL
    parser.add_argument('--multi_scale_model', type=str, choices = ['fpn', 'backbone_pyramid', 'msp'], default = None) 
    parser.add_argument('--scales', type=int,  nargs='*',  default=(16, 32, 64, 128), help="List of scales to use for the multi-scale model.")

    parser.add_argument('--fpn_dim', type=int, default=256)
    parser.add_argument('--fpn_in_channels', type=int, nargs='+', default=None,
                        help='Input channel sizes for FPN lateral convolutions (C4, C5). '
                             'Defaults to [120, 352] for B2; use [128, 176] for B5.')
    parser.add_argument('--upsample_method', type = str, choices = ['bilinear', 'nearest'], default = 'nearest')
    parser.add_argument('--norm_fpn', type = bool, default = False)
    
    parser.add_argument('--deep_supervision', action='store_true', default=False)
    parser.add_argument('--type_scale_aggregator', type=str, choices = ['concatenation', 'max_p', 'mean_p','attention', 'gated-attention'], default=None)

    #Nested MIL 
    parser.add_argument('--nested_model', action='store_true', default=False)
    parser.add_argument('--type_region_aggregator', type=str, choices = ['concatenation', 'max_p', 'mean_p','attention', 'gated-attention'], default=None)
    parser.add_argument('--type_region_encoder', default=None, choices=['mlp', 'sab', 'isab'], type=str, help="Type of MIL encoder.")
    parser.add_argument('--type_region_pooling', default=None, choices=['max', 'mean', 'attention', 'gated-attention', 'pma'], type=str, help="MIL pooling operator")
    
    # Training parameters
    parser.add_argument('--training_mode', type=str, default = 'frozen',
                        help = 'Training mode (default: pretrained)',
                        choices = ['finetune', 'frozen'])
    parser.add_argument('--warmup_stage_epochs', type = int, default = 0)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=9, type=int)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--weighted-BCE", default='n', type=str)
    parser.add_argument('--clip_grad', type=float, default=0.0, metavar='NORM',
                   help='Clip gradient norm (default: None, no clipping)')

    # Data augmentation settings
    parser.add_argument("--balanced-dataloader", default='n', type=str,help='Enable weighted sampling during training (default: False).')
    parser.add_argument("--data_aug", action='store_true', default=False)
    parser.add_argument(
        '--aug_config', default=None, type=str,
        help='Path to data_augmentation.yaml for feature-space augmentations '
             '(offline training only). If None, no feature augmentations are applied.'
    )
    
    parser.add_argument("--alpha", default=10, type=float)
    parser.add_argument("--sigma", default=15, type=float)
    parser.add_argument("--p", default=1.0, type=float)
    
    # LR scheduler settings 
    parser.add_argument("--lr", default=5.0e-5, type=float)
    parser.add_argument("--warmup-epochs", default=1, type=float)
    parser.add_argument("--epochs-warmup", default=0, type=float)
    parser.add_argument("--num_cycles", default=0.5, type=float)

    # Regularization parameters
    parser.add_argument('--drop_classhead', type=float, default=0.0, metavar='PCT', help='Dropout rate used in the classification head (default: 0.)')
    parser.add_argument('--drop_attention_pool', type=float, default=0.0, metavar='PCT', help='Dropout rate used in the attention pooling mechanism (default: 0.)')
    parser.add_argument('--drop_mha', type=float, default=0.0, metavar='PCT', help='Dropout rate used in the attention pooling mechanism (default: 0.)')
    parser.add_argument('--fcl_dropout', type=float, default=0.0)
    parser.add_argument("--lamda", type=float, default=0.0,
                        help='lambda used for balancing cross-entropy loss and rank loss.')
    
    # ROI evaluation parameters
    parser.add_argument('--roi_eval', action='store_true', default=False, help='Evaluate post-hoc detection performance')
    parser.add_argument('--roi_attention_threshold', type=float, default=0.5)
    parser.add_argument('--visualize_num_images', default=0, type=int, help="")
    parser.add_argument('--quantile_threshold', default = 0.95, type = float) 
    parser.add_argument('--max_bboxes', default = 100, type = int)
    parser.add_argument('--min_area', default = 1024, type = int)
    parser.add_argument('--iou_threshold', default = 0.25, type = float)

    parser.add_argument('--roi_eval_scheme', default='all_roi', choices = ['small_roi', 'medium_roi', 'large_roi', 'all_roi'], type=str, help="")
    parser.add_argument('--roi_eval_set', default='test', choices = ['val', 'test'], type=str, help="")

    parser.add_argument('--iou_method', default = 'iou', choices = ['iou', 'iobb_detection', 'iobb_annotation'], type = str) 
    parser.add_argument('--ap_method', default = 'area', choices = ['area', '11points'], type = str) 
    
    # Device settings 
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--apex", default="y", type=str)
    
    # Misc
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--print-freq", default=5000, type=int)
    parser.add_argument("--log-freq", default=1000, type=int)
    parser.add_argument("--running-interactive", default='n', type=str)
    parser.add_argument('--eval_scheme', default='kruns_train+val', type=str, help='Evaluation scheme [kruns_train+val | kfold_cv+test ]')
    parser.add_argument('--resume', default = None, type = str)
    parser.add_argument('--test_example', default = None, type = str)

    # wandb
    parser.add_argument('--wandb_project', default='msamil-vindr', type=str)
    parser.add_argument('--wandb_entity', default=None, type=str)
    parser.add_argument('--wandb_mode', default='online', choices=['online', 'offline', 'disabled'], type=str)
    parser.add_argument('--wandb_name', default=None, type=str, help='wandb run name (auto-generated if None)')

    return parser.parse_args()


def main(args):
    
    seed_all(args.seed) # Fix the seed for reproducibility
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # print('\ntorch.cuda.current_device():', torch.cuda.current_device())
    print('\nUsing device:', device)

    args.apex = True if args.apex == "y" else False
    
    # get paths
    now = datetime.now().strftime('%Y-%m-%d')
    
    args.running_interactive = True if args.running_interactive == "y" else False

    torch.cuda.empty_cache() # Clean up

    if args.train:

        # Load feature augmentation config BEFORE computing the experiment ID
        # so that its content is included in the hash.
        args.aug_config_dict = None
        if args.aug_config is not None:
            with open(args.aug_config, 'r') as _f:
                args.aug_config_dict = yaml.safe_load(_f)
            print(f"Feature augmentation config loaded from: {args.aug_config}")
        else:
            print("No feature augmentation config specified; running without feature augmentations.")

        # Capture git state before any file writes
        args.git_hash = _get_git_hash()
        print(f"git commit: {args.git_hash}")

        # Deterministic experiment ID: date prefix + 8-char hash of all meaningful args
        exp_id = _compute_exp_id(args, now.replace('-', ''))
        args.exp_id = exp_id
        args.wandb_group = args.wandb_name or exp_id
        args.wandb_tags = _build_wandb_tags(args)
        args.wandb_notes = (
            f"{args.dataset}/{args.label} | "
            f"model={args.multi_scale_model or 'single'} pool={args.pooling_type} "
            f"lr={args.lr} bs={args.batch_size} enc_dim={args.fcl_encoder_dim}"
        )

        # From MammoCLIP's work
        if args.weighted_BCE == "y" and args.dataset.lower() == "vindr" and args.label.lower() == "mass":
            args.BCE_weights = 15.573306370070778
        elif args.weighted_BCE == "y" and args.dataset.lower() == "vindr" and args.label.lower() == "suspicious_calcification":
            args.BCE_weights = 37.296728971962615

        if args.balanced_dataloader == 'y':
            print("[balanced_dataloader] WeightedRandomSampler enabled — minority class will be oversampled each epoch.")

        args.output_path = Path(args.output_dir) / "experiments" / args.dataset / args.label / exp_id
        args.registry_path = Path(args.output_dir) / "experiments_registry.csv"

        os.makedirs(args.output_path, exist_ok=True)
        print(f"output_path: {args.output_path}")

        # Persist augmentation config alongside the saved model
        if args.aug_config is not None:
            import shutil
            shutil.copy(args.aug_config,
                        os.path.join(args.output_path, 'data_augmentation.yaml'))

        # Save git hash as a standalone file so it's always findable next to the model
        with open(os.path.join(args.output_path, 'git_hash.txt'), 'w') as _f:
            _f.write(args.git_hash + '\n')

        # Convert any PosixPath in args.__dict__ to a string
        args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in args.__dict__.items()}

        # Cache the args as a text string to save them in the output dir
        args_text = yaml.safe_dump(args_dict, default_flow_style=False)

        with open(os.path.join(args.output_path, "args.yaml"), 'w') as f:
            f.write(args_text)

        do_experiments(args, device)

    elif args.evaluation:
        Eval(args, device)

    elif args.roi_eval:
        ROI_Eval(args, device)

if __name__ == "__main__":
    args = config()
    main(args)
