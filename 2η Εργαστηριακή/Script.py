
# coding: utf-8

# In[73]:


import numpy as np
import random
import scipy.special as sp
from pylab import *    # grid(True) -> Πλέγμα στα διαγράμματα

def plot_size(length, height):
    # Καθορισμός διαστάσεων διαγραμμάτων
    return plt.figure(figsize=(length, height))

bitstream = np.random.randint(2,size=18)
print bitstream

# Επιλεγμένο με την παραπάνω διαδικασία bitstream για λόγους συμβατότητας με τον συνάδελφο
bitstream = [0,1,0,1,1,1,1,0,0,1,0,1,0,0,1,1,0,1]
print bitstream


# In[74]:


# ===== Ερώτημα 1α =====

t_bit   = 1
v_pulse = 9   # ID 03115117 - ΑΒΡΑΜΙΔΗΣ ΚΛΕΑΝΘΗΣ
fs = 20       # Συχνότητα δειγματοληψίας

voltage = zeros(18*fs)   # fs δείγματα ανά sec / bit
y = zeros(18)            # 1 δείγμα ανά bit
for i in range(0,18):
    # BPAM: προσαρμογή πλάτους
    if (bitstream[i] == 0):
        y[i] = -v_pulse
        for j in range(0,fs): voltage[fs*i+j] = -v_pulse
    else:
        y[i] = v_pulse
        for j in range(0,fs): voltage[fs*i+j] = v_pulse

# Χρόνος 18 sec με 18*fs δείγματα => 1 sec ανά bit
x_axis = np.linspace(0, 18*t_bit, (18*fs)*t_bit)
plot_size(15,3)
grid(True)
plt.plot(x_axis, voltage)
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("Bit Stream ( BPAM )")
plt.show()


# In[75]:


# ===== Ερώτημα 1β =====

# Εφαρμογή της μεθόδου Gram-Schmidt για την γεωμετρική αναπαράσταση των σημάτων

# Δημιουργία των προβολών
s_1 = v_pulse*ones(fs)
s_2 = -s_1

energy  = v_pulse*v_pulse*t_bit
# Δημιουργία ορθοκανονικής βάσης
phi_1 = s_1/np.sqrt(energy)
phi_2 = 0

# Περιγράφουμε τις προβολές συναρτήσει της ορθοκανονικής βάσης
s11 = (1.0/fs)*sum(s_1*phi_1)
s12 = (1.0/fs)*sum(s_1*phi_2)
s21 = (1.0/fs)*sum(s_2*phi_1)
s22 = (1.0/fs)*sum(s_2*phi_2)

# Αστερισμός
c1 = s11 + 1j*s12
c2 = s21 + 1j*s22
C = [c1, c2]
X = [x.real for x in C]
Y = [x.imag for x in C]
plt.scatter(X,Y)
grid(True)
plt.xlabel("phi_1")
plt.ylabel("phi_2")
plt.title("BPAM Constellation")
plt.show()


# In[76]:


# ===== Ερώτημα 1γ =====

# Η συνάρτηση add_awgn παίρνει ως είσοδο ένα καθαρό σήμα και προσθέτει σε αυτό λευκό γκαουσιανό 
# θόρυβο με κατάλληλη πυκνότητα ισχύος, ώστε ο σηματοθορυβικός λόγος να πάρει την επιθυμητή τιμή SNR_dB

def add_awgn(x, SNR_dB):
    L = len(x)
    # Θόρυβος: μιγαδική μεταβλητή με Re,Im ανεξάρτητες κανονικές κατανομές
    noise = np.random.randn(1,L) + 1j*np.random.randn(1,L) 
    
    signal_power = sum(abs(x)*abs(x))/L
    noise_power = sum(abs(noise)*abs(noise))/L
    #Προσαρμογή της ισχύος του θορύβου στο επιθυμητό SNR
    K = (signal_power/noise_power)*(10**(-SNR_dB/10))
    z = sqrt(K)*noise
    
    return (x + z)

# Χρησιμοποιούμε την πραγματική μεταβλητή του θορύβου
noisy_bpam_1 = add_awgn(voltage, 5).T
plot_size(20,5)
grid(True)
plt.plot(x_axis, noisy_bpam_1.real)
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("Noisy BPAM signal (5 Eb/N0)")
plt.show()

noisy_bpam_2 = add_awgn(voltage, 15).T
plot_size(20,5)
grid(True)
plt.plot(x_axis, noisy_bpam_2.real)
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("Noisy BPAM signal (15 Eb/N0)")
plt.show()


# In[77]:


# ===== Ερώτημα 1δ =====

# Χρησιμοποιούμε 1 δείγμα ανά bit
rs1 = add_awgn(y,5).T
rs2 = add_awgn(y,15).T

#Δημιουργία αστερισμών
RE1 = [x.real for x in rs1]
IM1 = [x.imag for x in rs1]
plt.scatter(RE1,IM1)
grid(True)
plt.xlabel("phi_1")
plt.ylabel("phi_2")
plt.title("Noisy BPAM Constellation (5 Eb/N0)")
plt.show()

RE2 = [x.real for x in rs2]
IM2 = [x.imag for x in rs2]
plt.scatter(RE2,IM2)
grid(True)
plt.xlabel("phi_1")
plt.ylabel("phi_2")
plt.title("Noisy BPAM Constellation (15 Eb/N0)")
plt.show()


# In[78]:


# ===== Ερώτημα 1ε =====

# Δημιουργία κατάλληλου αριθμού bits προς διαμόρφωση
bit_number  = 5000000
bitstream_l = np.random.randint(2,size=bit_number)

# Διαμόρφωση των bits κατά BPAM (1 δείγμα ανά bit)
voltage_l   = zeros(bit_number)  
for i in range(0,bit_number):
    if (bitstream_l[i] == 0): voltage_l[i] = -v_pulse
    else: voltage_l[i] = v_pulse

# Ορισμός της συνάρτησης Q
def Q(x):
    return 0.5*sp.erfc(x/np.sqrt(2.0))

#Υπολογισμός της θεωρητικής και πειραματικής πιθανότητας σφάλματος κατά την μετάδοση για SNR από 0 ως 15 dB
ber = zeros(15)
theoretical = zeros(15)
for snr in range (0,15):
    #Προσθήκη λευκού θορύβου στο σήμα
    noisy_l = add_awgn(voltage_l, snr).T
    
    received_signal = zeros(bit_number) + 1j*zeros(bit_number)
    decision = zeros(bit_number)
    # Βρίσκουμε τα λάθος ψηφία βάσει προσήμου
    for i in range (0,bit_number):
        if (noisy_l[i].real > 0): decision[i] = 1
        else: decision[i] = 0
        
        if (decision[i] != bitstream_l[i]): ber[snr] = ber[snr] + 1.0/bit_number
    # Υπολογισμός θεωρητικής τιμής        
    snr_lin = 10**(snr/10)        
    theoretical[snr] = Q(np.sqrt(2*snr_lin))

plot_size(5,7)
grid(True)
plt.plot(ber, label='Experimental BER')
plt.plot(theoretical, label='Theoretical BER')
plt.ylim(10**(-6),10**(-1))
plt.xlim(0,15)
plt.yscale('log')
plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bir error rate")
plt.title("Bit Error Rate in BPAM")
plt.legend()
plt.show()


# In[79]:


# ===== Άσκηση 2 =====
# To ερώτημα α απαντάται στην αναφορά

t_bit = 1
v_max = 9
fs = 100     # Συχνότητα δειγματοληψίας (δείγματα ανά bit)
fc = 2       # ID 03115117
t  = np.arange(0,18,1.0/fs)

# Διαμόρφωση κατά BPSK
bpsk = zeros(18*fs)
for i in range (0, 18):
    if (bitstream[i] == 0): 
        for j in range (0, fs): bpsk[i*fs+j] =  v_max*np.sin(2*np.pi*fc*t[j])   # symbol = 0
    else:
        for j in range (0, fs): bpsk[i*fs+j] = -v_max*np.sin(2*np.pi*fc*t[j])   # symbol = 1

plot_size(20,3)
grid(True)
plt.xlim(0,18)
plt.plot(t,bpsk)
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("Bit Stream ( BPSK )")
plt.show()


# In[80]:


#Διαμόρφωση κατά QPSK (Grey Coding)
qpsk = zeros(18*fs)
for i in range (0,9):
    if (bitstream[2*i] == 0 and bitstream[2*i+1] == 0): 
        for j in range (0, 2*fs): qpsk[2*i*fs+j] = v_max*np.sin(2*np.pi*fc*t[j])              # symbol = 00
    elif (bitstream[2*i] == 0 and bitstream[2*i+1] == 1):
        for j in range (0, 2*fs): qpsk[2*i*fs+j] = v_max*np.sin(2*np.pi*fc*t[j] + np.pi/4)    # symbol = 01
    elif (bitstream[2*i] == 1 and bitstream[2*i+1] == 1):
        for j in range (0, 2*fs): qpsk[2*i*fs+j] = v_max*np.sin(2*np.pi*fc*t[j] + np.pi/2)    # symbol = 11
    else:
        for j in range (0, 2*fs): qpsk[2*i*fs+j] = v_max*np.sin(2*np.pi*fc*t[j] + 3*np.pi/4)  # symbol = 10

plot_size(20,3)
grid(True)
plt.xlim(0,18)
plt.plot(t,qpsk)
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("Bit Stream ( QPSK )")
plt.show()


# In[81]:


#Διαμόρφωση κατά 8-PSK (Grey Coding)
epsk = zeros(18*fs)
for i in range (0,6):
    if (bitstream[3*i] == 0 and bitstream[3*i+1] == 0 and bitstream[3*i+2] == 0):
        for j in range (0, 3*fs): epsk[3*i*fs+j] = v_max*np.sin(2*np.pi*fc*t[j])              # symbol = 000
    elif (bitstream[3*i] == 0 and bitstream[3*i+1] == 0 and bitstream[3*i+2] == 1):
        for j in range (0, 3*fs): epsk[3*i*fs+j] = v_max*np.sin(2*np.pi*fc*t[j] + np.pi/8)    # symbol = 001
    elif (bitstream[3*i] == 0 and bitstream[3*i+1] == 1 and bitstream[3*i+2] == 1):
        for j in range (0, 3*fs): epsk[3*i*fs+j] = v_max*np.sin(2*np.pi*fc*t[j] + 2*np.pi/8)  # symbol = 011
    elif (bitstream[3*i] == 0 and bitstream[3*i+1] == 1 and bitstream[3*i+2] == 0):
        for j in range (0, 3*fs): epsk[3*i*fs+j] = v_max*np.sin(2*np.pi*fc*t[j] + 3*np.pi/8)  # symbol = 010
    elif (bitstream[3*i] == 1 and bitstream[3*i+1] == 1 and bitstream[3*i+2] == 0):
        for j in range (0, 3*fs): epsk[3*i*fs+j] = v_max*np.sin(2*np.pi*fc*t[j] + 4*np.pi/8)  # symbol = 110
    elif (bitstream[3*i] == 1 and bitstream[3*i+1] == 1 and bitstream[3*i+2] == 1):
        for j in range (0, 3*fs): epsk[3*i*fs+j] = v_max*np.sin(2*np.pi*fc*t[j] + 5*np.pi/8)  # symbol = 111
    elif (bitstream[3*i] == 1 and bitstream[3*i+1] == 0 and bitstream[3*i+2] == 1):
        for j in range (0, 3*fs): epsk[3*i*fs+j] = v_max*np.sin(2*np.pi*fc*t[j] + 6*np.pi/8)  # symbol = 101
    else:
        for j in range (0, 3*fs): epsk[3*i*fs+j] = v_max*np.sin(2*np.pi*fc*t[j] + 7*np.pi/8)  # symbol = 100

plot_size(20,3)
grid(True)
plt.xlim(0,18)
plt.plot(t,epsk)
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("Bit Stream ( 8-PSK )")
plt.show()


# In[82]:


# ===== Ερώτημα 3α =====

in_phase = zeros(9)   #  Συμφασική συνιστώσα, Polar-NRZ για το 1ο ψηφίο των συμβόλων
quad = zeros(9)       # Ορθογωνική συνιστώσα, Polar-NRZ για το 2ο ψηφίο των συμβόλων
for i in range (0,9):
    if bitstream[2*i] == 0: in_phase[i] = -1
    else: in_phase[i] = 1
    
    if bitstream[2*i+1] == 0: quad[i] = -1
    else: quad[i] = 1

# Διαμορφωμένο σήμα
s = zeros(18*fs)
for i in range (0,9):
    for j in range (0,2*fs):
        s[2*fs*i+j] = in_phase[i]*v_max*np.cos(2*np.pi*fc*t[j]) + quad[i]*v_max*np.sin(2*np.pi*fc*t[j])

# Μέθοδος Gram-Schmidt

t = np.arange(0, 2*t_bit, 1.0/fs)
#Δημιουργία των προβολών
s_1 = v_max*np.cos(2*np.pi*fc*t - np.pi/4)   # 11
s_2 = v_max*np.cos(2*np.pi*fc*t + np.pi/4)   # 10
s_3 = v_max*np.cos(2*np.pi*fc*t - 3*np.pi/4) # 01
s_4 = v_max*np.cos(2*np.pi*fc*t + 3*np.pi/4) # 00

#Δημιουργία ορθοκανονικής βάσης
phi_1 = (v_max*np.cos(2*np.pi*fc*t)/np.sqrt(v_max*v_max*t_bit))
phi_2 = (v_max*np.sin(2*np.pi*fc*t)/np.sqrt(v_max*v_max*t_bit))

#Περιγράφουμε τις προβολές συναρτήσει της ορθοκανονικής βάσης
s11 = (1.0/fs)*sum(s_1*phi_1)
s12 = (1.0/fs)*sum(s_1*phi_2)
s21 = (1.0/fs)*sum(s_2*phi_1)
s22 = (1.0/fs)*sum(s_2*phi_2)
s31 = (1.0/fs)*sum(s_3*phi_1)
s32 = (1.0/fs)*sum(s_3*phi_2)
s41 = (1.0/fs)*sum(s_4*phi_1)
s42 = (1.0/fs)*sum(s_4*phi_2)

#Αστερισμός
c1 = s11 + 1j*s12
c2 = s21 + 1j*s22
c3 = s31 + 1j*s32
c4 = s41 + 1j*s42
C = [c1, c2, c3, c4]
X = [x.real for x in C]
Y = [x.imag for x in C]
plt.scatter(X,Y)
grid(True)
plt.xlabel("phi_1")
plt.ylabel("phi_2")
plt.title("Baseband QPSK Constellation")
plt.text(5.5,5.5,"00")
plt.text(-5.5,5.5,"01")
plt.text(-5.5,-5.5,"11")
plt.text(5.5,-5.5,"10")
plt.show()


# In[83]:


# ===== Ερώτημα 3β =====

# Το σήμα που λαμβάνει ο δέκτης αποτελείται από την ορθογωνική και 
# συμφασική συνιστώσα με προσθήκη λευκού θορύβου από τον δίαυλο
received_pure = v_max*(in_phase+1j*quad)
rs1 = add_awgn(received_pure,5).T
rs2 = add_awgn(received_pure,15).T

# Δημιουργία αστερισμών
RE1 = [x.real for x in rs1]
IM1 = [x.imag for x in rs1]
plt.scatter(RE1,IM1)
grid(True)
plt.xlabel("phi_1")
plt.ylabel("phi_2")
plt.title("Baseband noisy QPSK Constellation (5 Eb/N0)")
plt.show()

RE2 = [x.real for x in rs2]
IM2 = [x.imag for x in rs2]
plt.scatter(RE2,IM2)
grid(True)
plt.xlabel("phi_1")
plt.ylabel("phi_2")
plt.title("Baseband noisy QPSK Constellation (15 Eb/N0)")
plt.show()


# In[86]:


# ===== Ερώτημα 3γ =====

# Δημιουργία κατάλληλου αριθμού bits προς διαμόρφωση
bit_number  = 2000000
bitstream_l = np.random.randint(2,size=bit_number)  

# Διαμόρφωση κατά QPSK
in_phase = zeros(int(bit_number/2))
quad     = zeros(int(bit_number/2))
for i in range (0,int(bit_number/2)):
    if bitstream_l[2*i] == 0: in_phase[i] = -1
    else: in_phase[i] = 1
    
    if bitstream_l[2*i+1] == 0: quad[i] = -1
    else: quad[i] = 1
        
s = in_phase + 1j*quad

# Υπολογισμός της θεωρητικής και της πειραματικής πιθανότητας σφάλματος για SNR θορύβου από 0 ως 15 dB
ber = zeros(15)
theoretical = zeros(15)
for snr in range (0,15):
    counter1 = counter2 = 0
    noisy_l = add_awgn(s,snr).T
    
    # Βρίσκουμε ξεχωριστά για κάθε συνιστώσα τα λάθος ψηφία βάσει προσήμου
    # (θετικό --> 1 / αρνητικό --> -1)
    for i in range (0,int(bit_number/2)):
        if noisy_l[i].real>0:
            if in_phase[i]==-1: counter1 = counter1 + 1
        elif in_phase[i]==1: counter1 = counter1 + 1
               
    for i in range (0,int(bit_number/2)):
        if noisy_l[i].imag>0:
            if quad[i]==-1: counter2 = counter2 + 1
        elif quad[i]==1: counter2 = counter2 + 1

    ber[snr] = 0.5*(counter1+counter2)/bit_number    
    snr_new = 10**(snr/10)
    theoretical[snr] = Q(np.sqrt(2*snr_new))

plot_size(5,7)
grid(True)
plt.plot(ber, label='Experimental BER')
plt.plot(theoretical, label='Theoretical BER')
plt.ylim(10**(-6),10**(-1))
plt.xlim(0,12)
plt.yscale('log')
plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bir error rate")
plt.title("Bit Error Rate in QPSK")
plt.legend()
plt.show()

