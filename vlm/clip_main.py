from utils.utils import *
from train.logger import build_logger
from data.dataset import build_dataset
from train.clip_trainer import Trainer
from models.event_clip import build_model
from data.augmenter import build_augmenter
from train.scheduler import build_scheduler
from data.dataset import build_clip_dataset
from train.objectives import build_objective
from train.amp_scaler import build_amp_scaler
from pretrain.evaluator import build_evaluator
from train.checkpointer import build_checkpointer
from data.clip_dataloader import build_dataloader
from utils.config_parser import build_config_parser
from data.event_tokenizer import build_event_tokenizer

import torch.utils.data.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def main(rank: int,
         local_rank: int,
         world_size: int,
         resume: bool = False) -> None:

    # Set the experiment name
    experiment_name = 'train_clip'

    # Configure distributed training
    device = setup_distributed_environment(rank=rank,
                                           local_rank=local_rank,
                                           world_size=world_size)

    # Parse the configuration file
    parser = build_config_parser(config_file_path='configs/clip_config.yaml')
    config = parser.parse_config_file()

    # Build event augmenter
    event_augmenter = build_augmenter(config=config)

    # Prepare datasets for self-supervised learning
    train_dataset, val_dataset = build_clip_dataset(config=config,
                                                    transform=event_augmenter)

    # Create distributed samplers for self-supervised learning
    train_sampler = dist.DistributedSampler(dataset=train_dataset,
                                            num_replicas=world_size,
                                            rank=rank,
                                            shuffle=True) if world_size > 1 else None
    val_sampler = dist.DistributedSampler(dataset=val_dataset,
                                          num_replicas=world_size,
                                          rank=rank,
                                          shuffle=False) if world_size > 1 else None

    # Build an event tokenizer
    event_tokenizer = build_event_tokenizer(config=config,
                                            device=device)

    # Build dataloaders
    train_dataloader = build_dataloader(event_dataset=train_dataset,
                                        event_tokenizer=event_tokenizer,
                                        batch_size=config['data']['batch_size'] // world_size,
                                        num_workers=config['data']['num_workers'],
                                        local_rank=local_rank,
                                        shuffle=True,
                                        sampler=train_sampler,
                                        drop_last=True if world_size > 1 else False)

    val_dataloader = build_dataloader(event_dataset=val_dataset,
                                      event_tokenizer=event_tokenizer,
                                      batch_size=config['data']['batch_size'] // world_size,
                                      num_workers=config['data']['num_workers'],
                                      local_rank=local_rank,
                                      shuffle=False,
                                      sampler=val_sampler,
                                      drop_last=False)

    # Build the model on the corresponding device
    model = build_model(config=config,
                        device=device)

    # Print the number of trainable parameters
    print(f"The number of trainable parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}")

    # Build an optimizer, skip weight decay for biases and normalization layers,
    # set different learning rates for backbone and aligner
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=config['train']['optimizer']['lr'],
                                  weight_decay=config['train']['optimizer']['weight_decay'])

    # Wrap with DDP
    model = DDP(model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True) if world_size > 1 else model

    # Load the pretrained checkpoint
    model = load_pretrained_checkpoint(checkpoint_path=config['event_vision_model']['checkpoint'],
                                       model=model,
                                       input_backbone_name='backbone',
                                       output_backbone_name='event_encoder',
                                       device=device,
                                       verbose=True)

    # Define an objective function
    objective = build_objective(config=config)

    # Build a learning rate scheduler
    scheduler = build_scheduler(optimizer=optimizer,
                                config=config,
                                iter_per_epoch=len(train_dataloader))

    # Build a checkpointer
    checkpointer = build_checkpointer(checkpoint_dir=os.path.join(os.getcwd(), 'checkpoint'),
                                      model_config=config['event_vision_model'])

    # Load the checkpoint if resume
    if resume:
        checkpoint = checkpointer.load_checkpoint(device=device,
                                                  checkpoint_name=experiment_name)
        last_epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict = checkpoint

        # Load state dictionaries
        try:
            model.load_state_dict(model_state_dict, strict=False)
            optimizer.load_state_dict(optimizer_state_dict)

            if scheduler_state_dict is not None:
                scheduler.load_state_dict(scheduler_state_dict)
        except Exception as e:
            raise ValueError(f"Error loading state dictionaries: {str(e)}")
    else:
        last_epoch = 0

    # Define a logger
    logger = build_logger(log_dir=os.path.join(os.getcwd(), 'logs', experiment_name),
                          config=config,
                          resume=resume) if rank == 0 else None

    # Define an amp scaler
    amp_scaler = build_amp_scaler()

    # Define everything for linear evaluation
    lin_probe_train_dataset, lin_probe_val_dataset = build_dataset(data_path=config['data']['data_path'][0])
    lin_probe_train_sampler = dist.DistributedSampler(lin_probe_train_dataset,
                                                      num_replicas=world_size,
                                                      rank=rank,
                                                      shuffle=False) if world_size > 1 else None
    lin_probe_val_sampler = dist.DistributedSampler(lin_probe_val_dataset,
                                                    num_replicas=world_size,
                                                    rank=rank,
                                                    shuffle=False) if world_size > 1 else None
    evaluator = build_evaluator(model=model,
                                event_tokenizer=event_tokenizer,
                                train_dataset=lin_probe_train_dataset,
                                val_dataset=lin_probe_val_dataset,
                                train_sampler=lin_probe_train_sampler,
                                val_sampler=lin_probe_val_sampler,
                                rank=rank,
                                local_rank=local_rank)

    # Train the model
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      objective=objective,
                      scheduler=scheduler,
                      logger=logger,
                      evaluator=evaluator,
                      checkpointer=checkpointer,
                      amp_scaler=amp_scaler,
                      max_grad_norm=config['train']['max_grad_norm'],
                      rank=rank,
                      local_rank=local_rank,
                      last_epoch=last_epoch)
    trainer.train(train_dataloader=train_dataloader,
                  val_dataloader=val_dataloader,
                  num_epochs=config['train']['num_epochs'],
                  experiment_name=experiment_name)

    # Clean up distributed training
    cleanup_distributed_environment()


if __name__ == '__main__':
    # Obtain rank, local rank and world size from torchrun
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Run main() on the corresponding rank
    main(rank=rank,
         local_rank=local_rank,
         world_size=world_size,
         resume=False)
