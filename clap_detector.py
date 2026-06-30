
from pathlib import Path

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks

AUDIO_DIR = Path("audio")
FIGURES_DIR = Path("grafi")


def read_wav_mono(file_path):
    # Prebere WAV datoteko in jo pretvori v mono signal.

    fs, x = wavfile.read(file_path)

    # Če je signal stereo, vzamemo povprečje levega in desnega kanala.
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)

    # Če je bil signal zapisan kot integer, ga normaliziramo glede na največjo možno vrednost.
    # To naredi amplitudo primerljivo med različnimi posnetki.
    max_abs_value = np.max(np.abs(x))
    if max_abs_value > 0:
        x = x / max_abs_value

    # odstranimo enosmerno komponento, torej povprečno vrednost signala.
    x = x - np.mean(x)

    return fs, x


def highpass_filter(x, fs, cutoff_hz=700, order=4):
    """
    Uporabi visokoprepustni filter.
    Namen:
        Plosk ima veliko energije v višjih frekvencah.
        Govor, brnenje in počasne spremembe imajo več energije v nižjih frekvencah.
        S filtrom lahko zato nekoliko zmanjšamo vpliv nizkofrekvenčnega šuma.

    """

    nyquist = fs / 2

    # Če mejna frekvenca ni smiselna, vrnemo originalni signal.
    if cutoff_hz <= 0 or cutoff_hz >= nyquist:
        return x

    normalized_cutoff = cutoff_hz / nyquist

    # Oblikujemo Butterworthov visokoprepustni filter.
    b, a = butter(order, normalized_cutoff, btype="highpass")

    # filtfilt filtrira signal naprej in nazaj, zato ne uvede faznega zamika.
    y = filtfilt(b, a, x)

    return y


def short_time_energy(x, fs, frame_ms=20, hop_ms=10):
    """
    Izračuna kratkočasovno energijo signala.

    Signal razdelimo na kratka okna. Za vsako okno izračunamo povprečje kvadrata amplitude.
    Ploski so kratki in glasni dogodki, zato imajo v teh oknih veliko energijo.
    """

    frame_len = int(fs * frame_ms / 1000)
    hop_len = int(fs * hop_ms / 1000)

    # Varnostno preverjanje, da okna niso prekratka.
    frame_len = max(frame_len, 1)
    hop_len = max(hop_len, 1)

    energy = []
    times = []

    # Premikamo se po signalu z oknom dolžine frame_len.
    for start in range(0, len(x) - frame_len + 1, hop_len):
        end = start + frame_len
        frame = x[start:end]

        # Energija okna je povprečje kvadrata vzorcev.
        e = np.mean(frame ** 2)

        # Čas pripišemo sredini okna.
        t = (start + frame_len / 2) / fs

        energy.append(e)
        times.append(t)

    return np.array(times), np.array(energy)


def normalize_vector(v):
    """
    Normalizira vektor v območje [0, 1].
    To uporabimo za energijo, da je prag za zaznavanje lažje nastavljiv.
    """

    v_min = np.min(v)
    v_max = np.max(v)

    if v_max - v_min == 0:
        return np.zeros_like(v)

    return (v - v_min) / (v_max - v_min)


def smooth_signal(x, kernel_size=5):
    """
    Zgladi signal s preprostim drsečim povprečjem.
    Namen:
        Energija lahko med zaporednimi okni malo niha.
        Glajenje zmanjša majhna nihanja in ohrani večje vrhove, ki pripadajo ploskom.
    """

    if kernel_size <= 1:
        return x

    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(x, kernel, mode="same")


def detect_claps(energy, times, threshold=0.25, min_distance_s=0.25):
    #Zazna ploske kot vrhove v kratkočasovni energiji.
    

    # Energijo normaliziramo v območje [0, 1].
    energy_norm = normalize_vector(energy)

    # Energijo zgladimo, da zmanjšamo majhna nihanja.
    energy_norm = smooth_signal(energy_norm, kernel_size=5)

    # Izračunamo, koliko vzorcev na energijski časovni osi predstavlja minimalno razdaljo.
    if len(times) > 1:
        energy_fs = 1 / np.median(np.diff(times))
    else:
        energy_fs = 1

    min_distance_samples = int(min_distance_s * energy_fs)
    min_distance_samples = max(min_distance_samples, 1)

    # find_peaks poišče lokalne vrhove, ki so višji od praga.
    peaks, properties = find_peaks(
        energy_norm,
        height=threshold,
        distance=min_distance_samples
    )

    clap_times = times[peaks]

    return clap_times, peaks, energy_norm




def compute_spectrum(x, fs):
    #Izračuna frekvenčni spekter signala z FFT.


    # Uporabimo Hannovo okno, da zmanjšamo robne učinke pri FFT.
    window = np.hanning(len(x))
    x_windowed = x * window

    spectrum = np.fft.rfft(x_windowed)
    magnitude = np.abs(spectrum)

    freqs = np.fft.rfftfreq(len(x_windowed), d=1 / fs)

    # Normalizacija spektra za lažji prikaz.
    if np.max(magnitude) > 0:
        magnitude = magnitude / np.max(magnitude)

    return freqs, magnitude


def plot_waveform_with_claps(x, fs, clap_times, output_path):
    #Nariše časovni signal in označi zaznane ploske.
    

    t = np.arange(len(x)) / fs

    plt.figure(figsize=(12, 4))
    plt.plot(t, x, linewidth=0.8)

    # Vsak zaznan plosk označimo z navpično črto.
    for clap_time in clap_times:
        plt.axvline(clap_time, linestyle="--", linewidth=1.5)

    plt.title("Zvočni signal z označenimi zaznanimi ploski")
    plt.xlabel("Čas [s]")
    plt.ylabel("Amplituda")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_energy(times, energy_norm, clap_times, threshold, output_path):
    #Nariše normalizirano kratkočasovno energijo in prag zaznave.
    

    plt.figure(figsize=(12, 4))
    plt.plot(times, energy_norm, linewidth=1.2, label="Kratkočasovna energija")

    # Prag zaznave.
    plt.axhline(threshold, linestyle="--", linewidth=1.5, label="Prag zaznave")

    # Označimo zaznane ploske.
    for clap_time in clap_times:
        plt.axvline(clap_time, linestyle=":", linewidth=1.2)

    plt.title("Kratkočasovna energija signala")
    plt.xlabel("Čas [s]")
    plt.ylabel("Normalizirana energija")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_first_clap_spectrum(x, fs, clap_times, output_path, window_s=0.12):
    # Nariše frekvenčni spekter prvega zaznanega ploska.


    if len(clap_times) == 0:
        print("Spektra ploska ni mogoče narisati, ker ni bilo zaznanih ploskov.")
        return

    # Vzamemo prvi zaznan plosk.
    clap_time = clap_times[0]

    center = int(clap_time * fs)
    half_window = int((window_s / 2) * fs)

    start = max(center - half_window, 0)
    end = min(center + half_window, len(x))

    clap_segment = x[start:end]

    if len(clap_segment) < 2:
        print("Segment za spekter je prekratek.")
        return

    freqs, magnitude = compute_spectrum(clap_segment, fs)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, magnitude, linewidth=1.0)

    plt.title("Frekvenčni spekter prvega zaznanega ploska")
    plt.xlabel("Frekvenca [Hz]")
    plt.ylabel("Normalizirana amplituda")
    plt.xlim(0, min(10000, fs / 2))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def generate_demo_audio(output_path, fs=44100, duration_s=5.0):
    """
    Ustvari testni WAV posnetek s sintetičnimi ploski.
    Signal ni popolnoma realističen, je pa dovolj dober za preverjanje kode.
    """

    t = np.arange(int(duration_s * fs)) / fs

    # Osnovni tihi šum v ozadju.
    x = 0.01 * np.random.randn(len(t))

    # Časi sintetičnih ploskov.
    clap_times = [0.8, 2.1, 3.6, 4.4]

    for clap_time in clap_times:
        center = int(clap_time * fs)

        # Plosk modeliramo kot kratek šumni impulz z eksponentnim pojemanjem.
        clap_len = int(0.08 * fs)
        noise = np.random.randn(clap_len)
        envelope = np.exp(-np.linspace(0, 6, clap_len))
        clap = 0.8 * noise * envelope

        end = min(center + clap_len, len(x))
        valid_len = end - center

        if valid_len > 0:
            x[center:end] += clap[:valid_len]

    # Normalizacija pred zapisom.
    x = x / np.max(np.abs(x))

    # WAV datoteko zapišemo kot 16-bitni integer.
    x_int16 = np.int16(x * 32767)
    wavfile.write(output_path, fs, x_int16)

    print(f"Demo posnetek je shranjen v: {output_path}")


def analyze_file(
    input_path,
    threshold=0.25,
    min_distance_s=0.25,
    use_filter=True
):
   
   # Glavna funkcija za analizo ene WAV datoteke.


    FIGURES_DIR.mkdir(exist_ok=True)

    fs, x = read_wav_mono(input_path)

    # Za prikaz časovnega signala uporabimo originalni normaliziran signal.
    x_original = x.copy()

    # Za detekcijo lahko uporabimo filtriran signal.
    if use_filter:
        x_detection = highpass_filter(x, fs, cutoff_hz=700)
    else:
        x_detection = x

    # Izračun kratkočasovne energije.
    energy_times, energy = short_time_energy(
        x_detection,
        fs,
        frame_ms=20,
        hop_ms=10
    )

    # Zaznava ploskov.
    clap_times, peaks, energy_norm = detect_claps(
        energy,
        energy_times,
        threshold=threshold,
        min_distance_s=min_distance_s
    )

    # Izpis rezultatov.
    print()
    print("REZULTATI ANALIZE")
    print("-----------------")
    print(f"Datoteka: {input_path}")
    print(f"Frekvenca vzorčenja: {fs} Hz")
    print(f"Trajanje posnetka: {len(x_original) / fs:.2f} s")
    print(f"Število zaznanih ploskov: {len(clap_times)}")

    if len(clap_times) > 0:
        print("Časi zaznanih ploskov:")
        for i, clap_time in enumerate(clap_times, start=1):
            print(f"  {i}. plosk: {clap_time:.2f} s")
    else:
        print("Ni zaznanih ploskov. Poskusi z nižjim pragom, npr. --threshold 0.15")

    # Shranjevanje grafov.
    waveform_path = FIGURES_DIR / "waveform_detected_claps.png"
    energy_path = FIGURES_DIR / "short_time_energy.png"
    spectrum_path = FIGURES_DIR / "first_clap_spectrum.png"


    plot_waveform_with_claps(
        x_original,
        fs,
        clap_times,
        output_path=waveform_path
    )

    plot_energy(
        energy_times,
        energy_norm,
        clap_times,
        threshold=threshold,
        output_path=energy_path
    )

    plot_first_clap_spectrum(
        x_detection,
        fs,
        clap_times,
        output_path=spectrum_path
    )
    

    

    print()
    print("Shranjeni grafi:")
    print(f"  {waveform_path}")
    print(f"  {energy_path}")
    print(f"  {spectrum_path}")


    print()
    print("Shranjeni rezultati:")

    return clap_times


def main():
    """
    Glavna funkcija programa.
    Omogoča uporabo iz terminala.
    """

    parser = argparse.ArgumentParser(
        description="Zaznavanje ploskov v WAV zvočnem posnetku."
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="Pot do vhodne WAV datoteke, na primer audio/moj_posnetek.wav"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Prag za zaznavo ploskov po normalizaciji energije. Privzeto: 0.25"
    )

    parser.add_argument(
        "--min-distance",
        type=float,
        default=0.25,
        help="Najmanjša razdalja med dvema ploskoma v sekundah. Privzeto: 0.25"
    )

    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Izklopi visokoprepustni filter."
    )

    parser.add_argument(
        "--generate-demo",
        action="store_true",
        help="Ustvari demo WAV posnetek s sintetičnimi ploski."
    )

    args = parser.parse_args()

    AUDIO_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    # Če uporabnik izbere --generate-demo, ustvarimo testni posnetek.
    if args.generate_demo:
        demo_path = AUDIO_DIR / "demo_claps.wav"
        generate_demo_audio(demo_path)
        input_path = demo_path
    else:
        if args.input is None:
            print("Napaka: podaj WAV datoteko ali uporabi --generate-demo.")
            print("Primer:")
            print("  python clap_detector.py audio/moj_posnetek.wav")
            print("  python clap_detector.py --generate-demo")
            return

        input_path = Path(args.input)

    if not input_path.exists():
        print(f"Napaka: datoteka ne obstaja: {input_path}")
        return

    analyze_file(
        input_path=input_path,
        threshold=args.threshold,
        min_distance_s=args.min_distance,
        use_filter=not args.no_filter
    )


if __name__ == "__main__":
    main()