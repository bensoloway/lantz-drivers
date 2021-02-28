__version_info__=[1,5,0]
__CLIENT_VERSION__ = '.'.join(tuple(str(__version_info__[i]) for i in range(len(__version_info__))))

def _compare_version_number(version_1, version_2='.'.join(tuple(str(__version_info__[i]) if i==0 else str(__version_info__[i]+1) if i==1 else  '0' for i in range(len(__version_info__))))):
        #returns True, if fimrmware_version_1 is eaqual or higher than fimrware_version_2
        #if no second arument is given it compares the first argument with the next frimware version dedicated to update the current client software
        split_version_1 = version_1.split(' ')[0].split('.')
        split_version_2 = version_2.split(' ')[0].split('.')
        for i in range(max(len(split_version_1), len(split_version_2))):
            if i > len(split_version_1)-1:
                num1 = 0
                num2 = int(split_version_2[i])
            elif i >  len(split_version_2)-1:
                num1=int(split_version_1[i])
                num2=0
            else:
                num1=int(split_version_1[i])
                num2 = int(split_version_2[i])
            if num1 > num2:
                return 1
            elif num2 > num1:
                return -1
            else:
                continue
        return 0