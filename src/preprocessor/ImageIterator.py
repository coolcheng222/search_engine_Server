import os
import pickle
class ImageIterator:
    def __init__(self, root_dir, batch_size, checkpoint_file=None):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.folders = os.listdir(self.root_dir)
        self.current_folder_idx = 0
        self.current_image_idx = 0

        if checkpoint_file:
            self.load_checkpoint(checkpoint_file)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_folder_idx >= len(self.folders):
            raise StopIteration
        
        folder_path = self.folders[self.current_folder_idx]
        images = os.listdir(os.path.join(self.root_dir, folder_path))
        batch_images = []
        while len(batch_images) < self.batch_size:
            if self.current_image_idx >= len(images):
                self.current_image_idx = 0
                self.current_folder_idx += 1
                break
            
            image_path = os.path.join(folder_path, images[self.current_image_idx])
            batch_images.append(image_path)
            self.current_image_idx += 1
        
        if self.current_folder_idx < len(self.folders):
            self.save_checkpoint("checkpoint.pkl")
        
        return batch_images

    def save_checkpoint(self, checkpoint_file):
        checkpoint = {
            "current_folder_idx": self.current_folder_idx,
            "current_image_idx": self.current_image_idx
        }
        with open(checkpoint_file, "wb") as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            checkpoint = pickle.load(f)
        self.current_folder_idx = checkpoint["current_folder_idx"]
        self.current_image_idx = checkpoint["current_image_idx"]
