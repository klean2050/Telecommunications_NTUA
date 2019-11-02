
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from pylab import *    # grid(True) -> Πλέγμα στα διαγράμματα


# ID 03115117 - ΑΒΡΑΜΙΔΗΣ ΚΛΕΑΝΘΗΣ
fm  = 9000.0    
fs1 = fm * 20
fs2 = fm * 100
fs3 = fm * 5
A = 1  # Πλάτος Σήματος

def sample_time(f, fs, Tnum):
    # Διάνυσμα χρόνου μέχρι <Tnum> περιόδους με δείγμα ανά 1/fs (sec)
    return np.arange(0, Tnum * 1/f, 1/fs)

def signal(f, t):
    return A * np.sin(2*np.pi*f*t)

def plot_size(length, height):
    # Καθορισμός διαστάσεων διαγραμμάτων
    return plt.figure(figsize=(length, height))


# In[6]:


# Ερώτημα 1ο - Μέρος α΄

plot_size(15,5)
grid(True)
# Διάνυσμα Χρόνου
time1 = sample_time(fm,fs1,4)
# Κατασκευή διαγράμματος σήματος-χρόνου
plt.plot(time1, signal(fm,time1), 'o')
# Τίτλος-Υπόμνημα-Λεζάντες
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("(i) Sampling with fs1")

plot_size(15,5)
grid(True)
time2 = sample_time(fm,fs2,4)
plt.plot(time2, signal(fm,time2), 'o')
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("(ii) Sampling with fs2")

plot_size(15,5)
grid(True)
plt.plot(time2, signal(fm,time2), 'o', label='Sampling with fs2')
plt.plot(time1, signal(fm,time1), 'or', label='Sampling with fs1')
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("(iii) Shared Diagram")
plt.legend(loc=1)

plt.show()


# In[7]:


# Ερώτημα 1ο - Μέρος β΄

plot_size(15,5)
grid(True)
time3 = sample_time(fm,fs3,4)
plt.plot(time3, signal(fm,time3), 'o')
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("Sampling with 5*fm")

plt.show()


# In[8]:


# Ερώτημα 1ο - Μέρος γ΄

fm2  = fm + 1000
# z: Σύνθεση ταλαντώσεων παραπλήσιων συχνοτήτων
fosc = (fm+fm2)/2   # Συχνότητα ταλαντώσεων
fd   = fm2-fm       # Συχνότητα διακροτήματος
# Δειγματοληψία με βάση τη συχνότητα ταλαντώσεων
fs1 = fosc * 20
fs2 = fosc * 100
fs3 = fosc * 5

plot_size(15,10)
grid(True)
# Ζητείται η απεικόνιση 1 περιόδου διακροτήματος
tz1 = sample_time(fd,fs1,1)
# z = y + sin(2πt*fm2)
z1  = np.add(signal(fm,tz1), signal(fm2,tz1))
plt.plot(tz1, z1, 'o')
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("(i) Sampling with fs1")

plot_size(15,10)
grid(True)
tz2 = sample_time(fd,fs2,1)
z2  = np.add(signal(fm,tz2), signal(fm2,tz2))
plt.plot(tz2, z2, 'o')
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("(ii) Sampling with fs2")

plot_size(15,10)
grid(True)
plt.plot(tz2, z2, 'o', label='Sampling with fs2')
plt.plot(tz1, z1, 'or',label='Sampling with fs1')
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("(iii) Shared Diagram")
plt.legend(loc=1)


plot_size(15,10)
grid(True)
tz3 = sample_time(fd,fs3,1)
z3  = np.add(signal(fm,tz3), signal(fm2,tz3))
plt.plot(tz3, z3, 'o')
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("Sampling with 5*fd")

plt.show()


# In[9]:


# Ερώτημα 2ο - Μέρος α΄

# fm:3KHz
bits   = 4
levels = 2**bits
# Απόσταση επιπέδων κβάντισης
q     = 2.0/(levels-1)
delta = 2.0/levels

y = signal(fm, time1)
y_quantized = np.zeros(len(y))
# Βοηθητικός πίνακας
h = np.zeros(len(y))
for i in range(0, len(y)):
    h[i] = 2*(y[i]//q) + 1          # παίρνει τιμές -15, -13, ..., 13, 15
    y_quantized[i] = (delta/2)*h[i] # mid riser

plot_size(15,10)
grid(True)

# Διαμόρφωση κάθετου άξονα με τα επίπεδα κβαντισμού σε Natural Binary Coding
y_axis = np.linspace(-1+delta/2, 1-delta/2, 16)
y_axis_encoded=['0000','0001','0010','0011','0100','0101','0110','0111','1000','1001','1010','1011','1100','1101','1110','1111']
# Αντιστοίχιση των 16 τιμών του y_axis με τις NBC ετικέτες
plt.yticks(y_axis, y_axis_encoded)

plt.stem(time1, y_quantized)
plt.xlabel("Time (sec)")
plt.ylabel("Quantizing Levels ( Natural Binary Coding 4bit )")
plt.title("Quantized Signal")

plt.show()


# In[19]:


# Ερώτημα 2ο - Μέρος β΄

def SNR(y, y_quantized, length):
    # Σφάλμα κβάντισης για κάθε σημείο
    q_error = y_quantized - y
    
    # Υπολογισμός ισχύος σφάλματος
    error_power = 0
    for i in range(length):
        error_power += q_error[i]*q_error[i] / length
    print "Standard Deviation error for", length, "terms:", np.sqrt(error_power)
    
    # Υπολογισμός ισχύος σήματος
    signal_power = 0
    for i in range(length):
        signal_power += y[i]*y[i] / length     
    
    # SNR = ισχύς σήματος / ισχύς σφάλματος
    SNR_exp = signal_power/error_power
    SNR_exp_dB = 10*np.log10(SNR_exp)
    print "SNR for", length, "terms:", SNR_exp, "=", SNR_exp_dB, "dB\n"
    
SNR(y, y_quantized, 10) # 10 δείγματα
SNR(y, y_quantized, 20) # 20 δείγματα

#Υπολογισμός θεωρητικού SNR = ισχύς όλου του σήματος / θεωρητική διακύμανση
var_error_theory = delta**2 / 12
signal_power_theory = sum(y**2)/len(y)
SNR_theory = (signal_power_theory/var_error_theory)
SNR_theory_dB = 10*np.log10(SNR_theory)
print "Theoretical SNR: ", SNR_theory, "=", SNR_theory_dB, "dB" 


# In[12]:


# Ερώτημα 2ο - Μέρος γ΄

# Δημιουργία πίνακα bitstream
bit = zeros(80)
# 1 περίοδος => 20 δείγματα των 4bit => 80 bits
for i in range(0,20):
    level = y[i]//q
    temp = int(level+8)
    s = y_axis_encoded[temp] # αποθηκεύει το εκάστοτε level (4bit)
    # Μεταφορά των bits του επιπέδου στο bitstream
    bit[4*i]     = s[0]
    bit[4*i + 1] = s[1]
    bit[4*i + 2] = s[2]
    bit[4*i + 3] = s[3]


voltage = zeros(800)
# 1 bit αντιστοιχεί σε 10 δείγματα (800 για 80 bits)
v_pulse = fm/1000
for i in range(0,80):
    # POLAR NRZ: προσαρμογή πλάτους
    if (bit[i] == 0):
        for j in range(0,10): voltage[10*i+j] = -v_pulse
    else:
        for j in range(0,10): voltage[10*i+j] = v_pulse

# Χρόνος 0.08 sec με 800 δείγματα => 1 msec ανά bit
x_axis = np.linspace(0,0.08,800)
plot_size(15,5)
grid(True)
plt.plot(x_axis, voltage)
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("Bit Stream ( Polar NRZ )")

plt.show()


# In[15]:


# Ερώτημα 3ο - Μέρος α΄

t = sample_time(30.0, fs2, 4) # Διάνυσμα χρόνου
m = signal(30.0, t)           # Σήμα πληροφορίας
c = signal(fm, t)             # Φέρον σήμα
s = (1.0 + 0.5*m)*c           # Διαμορφωμένο ΑΜ σήμα με δείκτη 0.5

plot_size(20,5)
grid(True)
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("Modulated Signal")
plt.plot(t,s)

plt.show()


# In[16]:


# Eρώτημα 3ο - Μέρος β΄

#Αποδιαμόρφωση: διαμορφωμένο * φέρον => βαθυπερατό φίλτρο
to_filter = s*c
normalized_frequency = 30.0/fs2

n = 201 # Τάξη βαθυπερατού φίλτρου
# Συνάρτηση Μεταφοράς Φίλτρου: H(ω) = a(ω)/b(ω)
a = 1
b = ss.firwin(n, cutoff=normalized_frequency, window='blackmanharris')
filtered_data = ss.lfilter(b, a, to_filter)

#Διόρθωση της καθυστέρησης λόγω φίλτρου
delay = 200
v_out = filtered_data[delay:] 
t_new = t[0:(len(t)-delay)]
# Κάθε πολλαπλασιασμός cos υποδιπλασιάζει το πλάτος => *4
# Aφαιρούμε και τη dc συνιστώσα => τελική έξοδος
y_out = 4*(v_out - average(v_out))

plot_size(15,5)
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("Output compared to Input")
plt.plot(t_new, y_out, linewidth = 4, label = 'Demodulated Signal')
plt.plot(t_new, m[0:(len(m)-delay)], color = 'gold', linestyle = '--', label = 'Info Signal')
plt.grid(True)
plt.legend(loc=1)
plt.show()

