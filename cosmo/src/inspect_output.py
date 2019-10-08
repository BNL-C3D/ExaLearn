
from pathlib import Path

data_dir = Path("../output/")
cfg_dir  = Path("../cfg")
data = {} 
for f in data_dir.iterdir():
  if f.suffix == '.npy' and f.is_file():
    t = f.name.split('_')
#    print(t)
    if t[4] in data:
      if t[1:4] != data[t[4]][:3]: 
        print("ERROR: t[4]", t[1:4], data[t[4]][:3])
      else:
        data[t[4]][-1] += 1
    else:
        data[t[4]] = [*t[1:4], 1]
#    print(data)

print("Total data: ", len(data), sum( x[-1] for x in data.values()))

def print_feature(idx, name, data) :
  uniq_x  = sorted(set( x[idx] for x in data.values()))
  cnt = [sum( v[idx]==x for v in data.values()) for x in uniq_x ]
  cnt = list(map(str, cnt))
  fmt     = "{:>6}"*len(uniq_x) 
  print(name, fmt.format(*uniq_x))
  print("Counts:      ", fmt.format(*cnt))

print_feature( 0, "Omega-M      ", data)
print_feature( 1, "W0           ", data)
print_feature( 2, "sigma-8      ", data)

