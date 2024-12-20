from pathlib import Path
import argparse
import torch
import logging

##
# Save model checkpoint
# @param args: Command line arguments
# @param model: Model to save
# @param optimizer: Optimizer state
# @param epoch: Epoch number
def save_checkpoint(args, model, optimizer, epoch, iou_last):
    cptPath = Path(args.cfg)
    #cptName = "SAM_"+cptPath.stem+"_e"+str(epoch)+"_iou"+str(iou_last)+".pth"
    cptName = f"SAM_{cptPath.stem}_e{epoch}_iou{iou_last:.4f}.pth"
    logging.info(f"Saving checkpoint: {cptName}")
    
    cptDir = Path("models")
    cptDir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, cptDir/cptName)
