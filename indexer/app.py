import argparse

from web import get_web_app
from utils.logger import setup_logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help="dump index path", type=str, required=True)
    parser.add_argument('--device', help="pytorch device, e.g. cuda:0, cpu", type=str, default='cuda')
    parser.add_argument('--ip', help='ip address', type=str, default='0.0.0.0')
    parser.add_argument('--port', help='tcp port', type=int, default=8000)
    # parser.add_argument('--log', help='enable log', action='store_true')
    args = parser.parse_args()

    setup_logging('indexer.log')

    app = get_web_app(dict(
        dump_index_path=args.dir,
        device=args.device,
        index_mode='default',
        num_workers=4,
        batch_size=64,
    ))
        
    app.run(host=args.ip, port=args.port, debug=True)
