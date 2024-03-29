import argparse
from src.features import get_model, compute_features
from src.projections import compute_projections
from src.batches import create_batches
from src.defaults import defaults
from src.data import get_batch_dataset, get_label_mappings
from src.save import save_csv, save_backgrounds, save_scatter_plots

def compute_base(output_path, batches, base_id, model):
    print('Computing base features...')
    batch_dataset = get_batch_dataset(batches, base_id)
    features, path_images, predictions, labels = compute_features(model, batch_dataset)
    print('Computing base projections...')
    base_tsne, _ = compute_projections(features, path_images, predictions, labels, compute_base=True)
    return base_tsne

def extract_features(output_path, batches, model, label_mappings):
    base_id = defaults['base_tsne_id']
    base_tsne = compute_base(output_path, batches, base_id, model)
    
    print('Computing all features/projections...')
    num_batches = len(batches)
    for i in range(num_batches):
        batch_id = 'batch_{:04d}'.format(i + 1)
        batch_dataset = get_batch_dataset(batches, batch_id)
        features, path_images, predictions, labels = compute_features(model, batch_dataset)
        _, result_df = compute_projections(features, path_images, predictions, labels, base_tsne=base_tsne)
        csv_path = save_csv(output_path, batch_id, result_df)
        save_backgrounds(output_path, batch_id, csv_path)
        save_scatter_plots(output_path, batch_id, csv_path, label_mappings)
    print()



def main(dataset_path, model_path, output_path):
    label_mappings = get_label_mappings(dataset_path)
    num_classes, batches = create_batches(dataset_path, output_path)
    model = get_model(weights_path=model_path, num_classes=num_classes)
    extract_features(output_path, batches, model, label_mappings)
    print('Done!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot t-SNE from a trained model.')

    # Add the arguments
    parser.add_argument('--dataset_path', type=str, help='')
    parser.add_argument('--model_path', type=str, help='')
    parser.add_argument('--output_path', type=str, help='')

    # Parse the arguments
    args = parser.parse_args()
    print("Running!")
    main(args.dataset_path, args.model_path, args.output_path)