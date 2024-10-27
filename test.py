import io
import math
import torch
import random
import torch.nn as nn
from torch.optim import Adam
from pydub import AudioSegment
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
from test2 import fetch_view_serial, send_server_data, clear_client_data, view_client_data

device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"

ds_train = load_dataset("lewtun/music_genres", split="train")
ds_test = load_dataset("lewtun/music_genres", split="test")

ds = concatenate_datasets([ds_train, ds_test])
songs = ds.to_pandas()

songs['genre_id'] = songs['genre_id'].astype(int)


def convert_ogg_to_mp3(ogg_bytes):
    # Load ogg bytes as an AudioSegment
    audio = AudioSegment.from_file(io.BytesIO(ogg_bytes), format="ogg")

    # Export to mp3 in memory
    mp3_buffer = io.BytesIO()
    audio.export(mp3_buffer, format="mp3")

    # Get the mp3 bytes
    mp3_bytes = mp3_buffer.getvalue()

    return mp3_bytes


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class BPMModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_dim = 2

        self.positional_encoding = PositionalEncoding(d_model=self.model_dim)

        self.encoder_layers = nn.TransformerEncoderLayer(self.model_dim, 2, 256, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=4)

        self.sigmoid = nn.Sigmoid()

        self.decoder = nn.Sequential(
            nn.Linear(240, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(1), x.size(0), x.size(2))
        x = self.positional_encoding(x)
        x = x.view(x.size(1), x.size(0), x.size(2))
        x = self.encoder(x)

        x = x.view(x.size(0), -1)

        x = self.sigmoid(x)

        batch_size = x.size(0)

        padding_size = 240 - x.size(1)
        if padding_size > 0:
            padding = torch.full((batch_size, padding_size), -1, device=x.device)
            x = torch.cat((x, padding), dim=1)

        x = self.decoder(x)

        return x


model = BPMModel().to(device)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


class BPMDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sequence = self.data[idx]
        # Prepare input data
        inputs = torch.tensor(current_sequence[:-1], dtype=torch.float32)  # all but last
        targets = torch.tensor(current_sequence[-1][2], dtype=torch.float32)  # target is last element
        return inputs, targets


def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=-1.0)
    targets = torch.stack(targets)
    return inputs_padded, targets


def train_model(training_data):
    dataset = BPMDataset(training_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model.train()
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()


def main():
    bpm_history = []
    training_data = []
    current_genre = None
    sequence_length = 2
    random_song = None
    song_id = 0
    timestamp = 0
    while True:
        try:
            ser = fetch_view_serial()[-1]
            if ser['timestamp'] == timestamp:
                continue
            timestamp = ser['timestamp']
            bpm = int(ser['BPM'])
            print(f"BPM: {bpm}")
            if len(training_data) > 0:
                training_data[-1][-1] = (bpm / 240, current_genre / 18, bpm / 240)
                training_data = training_data[-1024:]
                train_model(training_data)
            bpm_history.append(bpm)
            bpm_history = bpm_history[-240:]

            # Minimize heart rate
            best_genre = None
            best_genre_bpm = 240
            model.eval()
            for genre_id in range(19):
                t_bpm_history = []
                for bpm_ in bpm_history:
                    t_bpm_history.append([bpm_ / 240, genre_id / 18])
                tensor_bpm_history = torch.tensor(t_bpm_history, device=device).unsqueeze(0)
                model_pred = model(tensor_bpm_history)
                predicted_resulting_bpm = model_pred[0].item() * 240
                # predicted_resulting_bpm = 90
                if predicted_resulting_bpm < best_genre_bpm:
                    best_genre_bpm = predicted_resulting_bpm
                    best_genre = genre_id

            # if current_genre != best_genre and best_genre is not None:
            if best_genre is not None:
                current_genre = best_genre
                if len(bpm_history) >= sequence_length:
                    if len(bpm_history) > sequence_length:
                        sequence_length = min(240, len(bpm_history))
                    sequence = [(bpm / 240, current_genre / 18) for bpm in bpm_history[-sequence_length:]]
                    sequence.append((bpm / 240, current_genre / 18, 0))  # Last element for label update
                    training_data.append(sequence)

                filtered_set = songs[songs['genre_id'] == int(current_genre)]
                filtered_set = filtered_set[filtered_set['song_id'] <= 2000]

                if not filtered_set.empty:
                    random_song = random.choice(filtered_set.to_dict(orient='records'))

                    # Check if the song_id is valid (<= 2000)
                    while random_song['song_id'] > 2000:
                        filtered_set = songs[songs['genre_id'] == int(current_genre)]
                        filtered_set = filtered_set[filtered_set['song_id'] <= 2000]
                        if not filtered_set.empty:
                            random_song = random.choice(filtered_set.to_dict(orient='records'))
                        else:
                            random_song = None
                            break

            client_code = 0
            client_data = view_client_data()
            if 'code' in client_data:
                client_code = client_data['code']

            # if random_song is not None and client_code == 14:
            if random_song is not None:
                song_id = random_song['song_id']
                # ogg_bytes = random_song['audio']['bytes']
                # mp3_bytes = convert_ogg_to_mp3(ogg_bytes)
                send_server_data(song_id)
                # send_server_data(0)
                clear_client_data()
                print("Skipping Song")
                # SEND DA SONG OVER BITCH
            else:
                if random_song is None:
                    print("random song none")

        except Exception as e:
            print(f"Exception: {e}")
            # Skip lines that don't have two valid float values
            pass


main()
