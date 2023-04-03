import argparse

from web import get_web_app

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help="dump index path", type=str, required=False, default='index')
    parser.add_argument('--ip', help='ip address', type=str, default='0.0.0.0')
    parser.add_argument('--port', help='tcp port', type=int, default=8000)
    # parser.add_argument('--log', help='enable log', action='store_true')
    args = parser.parse_args()

    app = get_web_app(args)
        
    app.run(host=args.ip, port=args.port, debug=True)
