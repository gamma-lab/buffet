# A shared memory approach for reducing preprocessing time in data analysis

The codes demonstrate how to cache preprocessed data in physical memory that can be later used for analysis and modeling. 



**Modules**

1. Loader: Read from disk -> Preprocess -> retain in interprocess shared memory -> exit
2. Reader: Read preprocessed data from shared memory -> perform modeling -> exit



**Memory**

1. The shared memory is retained till system reboot or unlinked by all used processes
2. Data should support serialization to Bytes Object



**API**

```
# Save object in shared physical memory
loader/save_to_shared_memory(obj, name)
	obj  := 	Any bytes serializable object
	name :=		Storage unit name (used in reader)
	
# Release process from using the object 
loader/release_process_from_shared_memory(name)
	name := 	Storage unit name 
	
# Load (read-only copy of) object from memory
reader/read_from_shared_memory(name)
	name := 	Storage unit name
```



**Setup**

Install libraries in `requirements.txt` in Python 3 environment

This particular demonstration requires Unix/Linux system. It can be extended to Windows.