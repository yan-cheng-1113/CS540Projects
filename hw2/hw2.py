import sys
import math
import string


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X = dict.fromkeys(string.ascii_uppercase, 0)
    with open (filename,encoding='utf-8') as f:
        # TODO: add your code here
        for line in f:
            text = line.strip().upper()
            for char in text:
                if char in X :
                    X[char] += 1

    return X



# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!
def main():
    shredder = shred('letter.txt')
    print('Q1')
    for key in shredder:
        print(key + " " + str(shredder[key]))

    print('Q2')
    esTuple = get_parameter_vectors()
    q_2e = shredder['A'] * math.log(esTuple[0][0])
    q_2s = shredder['A'] * math.log(esTuple[1][0])
    print(f'{q_2e : .4f}')
    print(f'{q_2s : .4f}')

    print('Q3')
    P_e = 0.6
    P_s = 0.4
    F_e = math.log(P_e)
    F_s = math.log(P_s)
    i = 0
    for v in shredder.values():
        F_e += v * math.log(esTuple[0][i])
        F_s += v * math.log(esTuple[1][i])
        i += 1
    print(f'{F_e:.4f}')
    print(f'{F_s:.4f}')

    print('Q4')
    P_eX = 1 / (1 + (math.e) ** (F_s - F_e))
    print(f'{P_eX:.4f}')
    

if __name__ == "__main__":

    main()


