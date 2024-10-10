import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


for exp in ['adiabatic', "isothermal"]:
    df = pd.read_csv(f"{exp}_data.csv")
    volumes = df['V_f (m^3)']
    works = df['Work (J)']

    plt.plot(volumes, works, label = exp)

plt.xlabel(r'final volume ($m^3$)')
plt.ylabel('work (J)')
plt.title('Work Done During Expansion vs Final Volume')
plt.legend()

plt.tight_layout()
plt.savefig('work_comparison.png', format = 'png')