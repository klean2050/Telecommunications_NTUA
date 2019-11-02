
# coding: utf-8

# In[121]:


import scipy.io.wavfile as wavf
import matplotlib.pylab as plt
import numpy as np

# ΚΛΕΑΝΘΗΣ ΑΒΡΑΜΙΔΗΣ
# ΑΜ: 03115117
# Άθροισμα τριών τελευταίων ψηφίων: 9 (περιττός)

# ===== Ερώτημα α =====

# Ανάγνωση των δεδομένων και του ρυθμού δειγματοληψίας από το αρχείο ήχου
[rate, data] = wavf.read("soundfile1_lab3.wav")
duration = 1.0*len(data)/rate
t = np.arange(0,duration,1.0/rate)

plt.figure(figsize=(20,10))
plt.grid(True)
plt.plot(t,data)
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude")
plt.title("Original audio signal")
plt.show()


# In[122]:


# ===== Ερώτημα β =====

# Κβάντιση του σήματος με ομοιόμορφο κβαντιστή 8 bits
bits   = 8
levels = 2**bits  # Αριθμός επιπέδων κβάντισης
q  = (np.amax(data)-np.amin(data))/(levels-1)   # Απόσταση επιπέδων κβάντισης

yq = np.zeros(len(data))
for i in range(0, len(data)):
    yq[i] = q*np.round(data[i]//q)

plt.figure(figsize=(20,10))
plt.grid(True)
plt.scatter(t, data, label='Original signal')
plt.plot(t, yq, color='orange', label='Quantized signal')
plt.xlabel("Time (sec)")
plt.ylabel("Quantizing Levels")
plt.title("Quantized Signal")
plt.legend()
plt.show()


# In[123]:


# Δημιουργία bitstream μέσω της αντιστοίχησης των επιπέδων κβάντισης σε δυαδικούς αριθμούς των 8 bits
bit = np.zeros(bits*len(yq))
for i in range(0,len(yq)):
    level = yq[i]//q
    s = format(int(level+levels/2),'08b')
    for j in range(0,bits):
        bit[bits*i+j] = int(s[j])


# In[124]:


# ===== Ερώτημα γ =====

# Διαμόρφωση κατά Baseband QPSK (Grey Coding)

num = int(len(bit)/2)
in_phase = np.zeros(num)   #  Συμφασική συνιστώσα, Polar-NRZ για το 1ο ψηφίο των συμβόλων
quad = np.zeros(num)       # Ορθογωνική συνιστώσα, Polar-NRZ για το 2ο ψηφίο των συμβόλων
for i in range (0,num):
    if bit[2*i] == 0: in_phase[i] = -1
    else: in_phase[i] = 1
    
    if bit[2*i+1] == 0: quad[i] = -1
    else: quad[i] = 1

signal = in_phase+1j*quad


# In[125]:


# ===== Ερώτημα δ =====

# Η συνάρτηση add_awgn παίρνει ως είσοδο ένα καθαρό σήμα και προσθέτει σε αυτό λευκό γκαουσιανό θόρυβο με κατάλληλη
# πυκνότητα φάσματος ισχύος, ώστε ο σηματοθορυβικός λόγος να πάρει την επιθυμητή τιμή SNR_dB

def add_awgn(x, SNR_dB):
    L = len(x)
    # Θόρυβος: μιγαδική μεταβλητή με Re,Im ανεξάρτητες κανονικές κατανομές
    noise = np.random.randn(L) + 1j*np.random.randn(L) 
    
    signal_power = sum(abs(x)*abs(x))/L
    noise_power = sum(abs(noise)*abs(noise))/L
    
    # Προσαρμογή της ισχύος του θορύβου στο επιθυμητό SNR
    K = (signal_power/noise_power)*(10**(-SNR_dB/10))
    z = np.sqrt(K)*noise
    
    return (x + z)

# Το σήμα που λαμβάνει ο δέκτης αποτελείται από την ορθογωνική και 
# συμφασική συνιστώσα με προσθήκη λευκού θορύβου από τον δίαυλο
received_pure = signal
rs1 = add_awgn(received_pure,5.0).T
rs2 = add_awgn(received_pure,15.0).T


# In[126]:


# ===== Ερώτημα ε =====

# Η αποδιαμόρφωση γίνεται σε δύο στάδια: απόφαση και αποκωδικοποίηση

# Υλοποίηση κυκλώματος απόφασης για την QPSK:
# Η απόφαση για κάθε bit γίνεται βάσει του τεταρτημορίου στο οποίο βρίσκεται η γεωμετρική του αναπαράσταση μετά την προσθήκη
# του λευκού θορύβου

def qpsk_decision(x):
    bit_dec = np.zeros(2*len(x))
    for i in range (0,len(x)):
        if (x[i].real > 0) and (x[i].imag > 0):
            bit_dec[2*i] = 1
            bit_dec[2*i+1] = 1
        elif (x[i].real > 0) and (x[i].imag < 0):
            bit_dec[2*i] = 1
            bit_dec[2*i+1] = 0
        elif (x[i].real < 0) and (x[i].imag < 0):
            bit_dec[2*i] = 0
            bit_dec[2*i+1] = 0
        else:
            bit_dec[2*i] = 0
            bit_dec[2*i+1] = 1
    return bit_dec

decision1 = qpsk_decision(rs1)
decision2 = qpsk_decision(rs2)


# In[127]:


# Η συνάρτηση decode λάμβανει ως είσοδο μία σειρά συμβόλων x, στην οποία κάθε σύμβολο αποτελείται από έναν αριθμό bits.
# Κάθε σύμβολο αντιστοιχίζεται σε ένα επίπεδο κβάντισης και στην συνέχεια πολλαπλασιάζεται με το βήμα κβάντισης, ώστε να
# ανακτηθεί το αρχικό κβαντισμένο σήμα

def decode(x, bits):   
    # Μέγεθος συμβόλου
    symbols = len(x)//bits
    # Αριθμός επιπέδων κβάντισης
    levels = 2**bits
    
    # Ο πίνακας decoded περιέχει το αποκωδικοποιημένο σήμα
    decoded = np.zeros(symbols)
    unsigned_8_bit = np.zeros(symbols)
    for i in range (0, symbols):
        # Αντιστοίχηση συμβόλου με επίπεδο κβάντισης
        dec = 0
        for j in range (0, bits): dec = dec*2 + x[bits*i+j] 
        # Πολλαπλασιασμός με βήμα κβάντισης (μας ενδιαφέρει η unsigned απεικόνιση των επιπέδων)
        decoded[i] = dec*q
    
    return decoded

decoded1 = decode(decision1, 8)
decoded2 = decode(decision2, 8)


# In[128]:


# Μέθοδος Gram-Schmidt για τους αστερισμούς

# Δημιουργία αστερισμών
RE1 = [x.real for x in rs1]
IM1 = [x.imag for x in rs1]
plt.scatter(RE1,IM1, marker='.')
plt.xlabel("In Phase")
plt.ylabel("Quadrate")
plt.title("Baseband noisy QPSK Constellation (5 Eb/N0)")
plt.show()

RE2 = [x.real for x in rs2]
IM2 = [x.imag for x in rs2]
plt.scatter(RE2,IM2, marker='.')
plt.xlabel("In Phase")
plt.ylabel("Quadrate")
plt.title("Baseband noisy QPSK Constellation (15 Eb/N0)")
plt.show()


# In[129]:


# ===== Ερώτημα στ =====

# Υπολογισμός της θεωρητικής και της πειραματικής πιθανότητας σφάλματος για SNR θορύβου 5 & 15 dB
import scipy.special as sp

# Ορισμός της συνάρτησης Q
def Q(x):
    return 0.5*sp.erfc(x/np.sqrt(2.0))

def ber(x,snr):
    counter1 = counter2 = 0
    # Βρίσκουμε ξεχωριστά για κάθε συνιστώσα τα λάθος ψηφία βάσει προσήμου
    # (θετικό --> 1 / αρνητικό --> -1)
    for i in range (0,num):
        if x[i].real>0:
            if in_phase[i]==-1: counter1 = counter1 + 1
        elif in_phase[i]==1: counter1 = counter1 + 1
               
    for i in range (0,num):
        if x[i].imag>0:
            if quad[i]==-1: counter2 = counter2 + 1
        elif quad[i]==1: counter2 = counter2 + 1

    ber = 0.5*(counter1+counter2)/len(bit)   
    snr_new = 10**(snr/10.0)
    theoretical = Q(np.sqrt(2*snr_new))
    return ber, theoretical

[ber5, theor5]   = ber(rs1,5)
[ber15, theor15] = ber(rs2,15)

print ("Bit Error rate for Eb/N0 = 5 dB:")
print "Experimental:", ber5, "Theoretical:", theor5, '\n'

print ("Bit Error rate for Eb/N0 = 15 dB:")
print "Experimental:", ber15, "Theoretical:", theor15


# In[130]:


# ===== Ερώτημα ζ =====

# Μετατρέπουμε τα αποκωδικοποιημένα σήματα σε μορφή unsigned int 8 bits 
unsign_int8_1 = np.uint8(decoded1/np.max(np.abs(decoded1))*255)
unsign_int8_2 = np.uint8(decoded2/np.max(np.abs(decoded2))*255)

# Εγγραφή των νέων αρχείων ήχου
wavf.write("new_sound1_03115117.wav",rate,unsign_int8_1)
wavf.write("new_sound2_03115117.wav",rate,unsign_int8_2)

