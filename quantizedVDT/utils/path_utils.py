import getpass
import socket


def get_abailoni_hci_home_path():
    username = getpass.getuser()
    hostname = socket.gethostname()
    if hostname == 'trendytukan' and username == 'abailoni':
        return '/net/hciserver03/storage/abailoni/'
    elif hostname == 'trendytukan' and username == 'abailoni_local':
        return '/home/abailoni_local/hci_home/'
    elif (hostname == 'ialgpu01' or hostname == 'birdperson' or hostname == 'sirherny') and username == 'abailoni':
        return '/home/abailoni/hci_home/'
    elif hostname == 'quadxeon5' and username == 'abailoni':
        return '/srv/scratch/abailoni'
    else:
        return '/net/hciserver03/storage/abailoni/'

def get_trendytukan_drive_path():
    username = getpass.getuser()
    hostname = socket.gethostname()
    # print(username, hostname)
    if hostname == 'trendytukan' and username == 'abailoni':
        return '/mnt/localdata0/abailoni/'
    elif hostname == 'trendytukan' and username == 'abailoni_local':
        return '/home/abailoni_local/trendyTukan_localdata0/'
    elif (hostname == 'ialgpu01' or hostname == 'birdperson' or hostname == 'sirherny') and username == 'abailoni':
        return '/home/abailoni/trendyTukan_drive/'
    elif hostname == 'quadxeon5' and username == 'abailoni':
        return '/srv/scratch/abailoni'
    else:
        raise NotImplementedError("Trendytukan local drive not accessible by the current user")
        # return '/net/hciserver03/storage/abailoni/trendyTukan_drive/'
