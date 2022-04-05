from model import Network
import torch
from phonemes import PHONEME_MAP
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from ctcdecode import CTCBeamDecoder

from dataset import LibriSamplesTest


def test(test_loader, decoder, device='cuda', model_path = './checkpoint/val_46.63.pth'):
    model = Network().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, leave=False, position=0, desc='Test')

    preds = []
    for x, len_x in test_loader:
        x = x.cuda()
        
        with torch.no_grad():
            outputs, out_lengths = model(x, len_x)
            beam_results, _, _, out_len = decoder.decode(outputs.permute(1, 0, 2), seq_lens=out_lengths)
            pred = "".join(PHONEME_MAP[j] for j in beam_results[0, 0, :out_len[0, 0]])
            preds.append(pred)

        batch_bar.update()   

    result = []
    for idx, pred in enumerate(preds):
        result.append([idx, pred])
    df = pd.DataFrame(result, columns=['id', 'predictions'])
    df.to_csv('./submission/submission_{:.02f}.csv'.format(model_path.split("/")[-1].split(".")[0]), index=False)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    root = "./hw3p2_student_data/hw3p2_student_data/"
    test_data = LibriSamplesTest(root, 'test_order.csv')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1, collate_fn=LibriSamplesTest.collate_fn)
    decoder = CTCBeamDecoder(labels=PHONEME_MAP, log_probs_input=True) # TODO: Intialize the CTC beam decoder   
    
    test(test_loader, decoder, device=device)