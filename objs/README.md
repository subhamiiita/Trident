In order to load these files:
<br/>
run the follwing command to load the data:
<br/>
```
import helpers
movie_enc= helpers.load_pkl("aud_64.obj")

movie_shape= len(movie_enc.get(list(movie_enc.keys())[0]))
```
