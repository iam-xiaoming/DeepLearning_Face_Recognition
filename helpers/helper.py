import torch

def save_checkpoint(model, margin, epoch, optimizer, avg_train_loss, test_loss, tar_far_3, tar_far_4):
    model_to_save = model.module if hasattr(model, 'module') else model
    margin_to_save = margin.module if hasattr(margin, 'module') else margin
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'margin_state_dict': margin_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'test_loss': test_loss,
        'tar_far1e-3': tar_far_3,
        'tar_far1e-4': tar_far_4,
    }, f'checkpoint_epoch_{epoch+1}.pt')
    print(f"Saved checkpoint at epoch {epoch+1}")
    
    
def print_results(optimizer, epoch, NUM_EPOCHS, avg_train_loss, test_loss, eval_res):
    current_lr_backbone = optimizer.param_groups[0]['lr']
    current_lr_margin = optimizer.param_groups[1]['lr']
    
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"Learning Rate: Backbone={current_lr_backbone:.6f}, Margin={current_lr_margin:.6f}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Eval Metrics:")
    print(f"  - Accuracy: {eval_res['accuracy']:.4f}")
    print(f"  - ROC AUC: {eval_res['roc_auc']:.4f}")
    print(f"  - TAR@FAR1e-3: {eval_res['tar_far_3']:.4f}")
    print(f"  - TAR@FAR1e-4: {eval_res['tar_far_4']:.4f}")
    print(f"  - Threshold: {eval_res['threshold']:.4f}")
    print(f"{'='*60}\n")