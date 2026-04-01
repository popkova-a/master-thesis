from utils.utils import *
from data.dataset import build_dataset
from models.event_clip import build_model
from utils.config_parser import build_config_parser
from data.event_tokenizer import build_event_tokenizer

from torch.utils.data import DataLoader
import torch.utils.data.distributed as dist
from GET_Transformer.models.GET import GET


def main(rank: int,
         local_rank: int,
         world_size: int,
         resume: bool = False) -> None:

    # Set the experiment name
    experiment_name = 'clip'

    # Configure distributed training
    device = setup_distributed_environment(rank=rank,
                                           local_rank=local_rank,
                                           world_size=world_size)

    # Parse the configuration file
    parser = build_config_parser(config_file_path='configs/clip_config.yaml')
    config = parser.parse_config_file()

    # Prepare datasets
    train_dataset, val_dataset = build_dataset(data_path=config['data']['data_path'][0])

    # Create a distributed samplers
    train_sampler = dist.DistributedSampler(dataset=train_dataset,
                                            num_replicas=world_size,
                                            rank=rank,
                                            shuffle=True) if world_size > 1 else None
    val_sampler = dist.DistributedSampler(dataset=val_dataset,
                                          num_replicas=world_size,
                                          rank=rank,
                                          shuffle=False) if world_size > 1 else None

    # Build event tokenizer
    event_tokenizer = build_event_tokenizer(config=config,
                                            device=device)

    # Build dataloaders
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config['data']['batch_size'] // world_size,
                                  num_workers=8,
                                  collate_fn=lambda batch: [[sample[0] for sample in batch],
                                                            [sample[1] for sample in batch]],
                                  shuffle=True if train_sampler is None else False,
                                  sampler=train_sampler,
                                  drop_last=True,
                                  persistent_workers=True,
                                  pin_memory=False)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=config['data']['batch_size'] // world_size,
                                num_workers=8,
                                collate_fn=lambda batch: [[sample[0] for sample in batch],
                                                          [sample[1] for sample in batch]],
                                shuffle=False,
                                sampler=val_sampler,
                                drop_last=True,
                                persistent_workers=True,
                                pin_memory=False)

    model = build_model(config=config,
                        device=device)

    # Load weights
    checkpoint = torch.load('/data/storage/anastasia/repos/master_thesis_anastasia_popkova/generative_vlm/checkpoint/train_clip_all.pth.tar',
                            map_location=device)
    model_state_dict = checkpoint.get('model_state_dict')
    model_state_dict = {k.replace('module.', '').replace('backbone.', ''): v for k, v in model_state_dict.items() if 'head' not in k}
    model.load_state_dict(model_state_dict, strict=False)
    model.cuda()

    # Freeze the backbone
    for p in model.parameters():
        p.requires_grad = False

    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2,
                perplexity=30,
                n_iter=3000,
                learning_rate='auto',
                init='pca',
                random_state=42)

    features = []
    labels = []
    from tqdm import tqdm
    i=0
    for batch in tqdm(train_dataloader):
        event_batch = batch[0]
        label_batch = batch[1]
        event_representations = torch.stack([event_tokenizer(events.unsqueeze(0)).squeeze(0) for events in event_batch],
                                            dim=0)
        features.append(model.extract_features(event_representations.unsqueeze(1)))
        labels.extend(label_batch)
        i+=1
        #if i>15:
            #break


    features = torch.cat(features, dim=0)
    #labels = torch.cat(labels, dim=0)
    features_2d = tsne.fit_transform(features.cpu().detach().numpy())

    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', s=10)

    plt.colorbar(scatter, ticks=range(10), label='Class')
    plt.title('t-SNE Visualization of Frozen Encoder Features')
    plt.tight_layout()

    # Save image
    plt.savefig("tsne_visualization.png", dpi=300)
    print("Saved t-SNE plot to tsne_visualization.png")


if __name__ == '__main__':
    # Obtain rank and world size from torchrun
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Run main() on the corresponding rank
    main(rank=rank,
         local_rank=local_rank,
         world_size=world_size,
         resume=False)
