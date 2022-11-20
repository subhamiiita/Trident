In order to load these files:
<br/>
run the follwing command to load the data:
<br/>
```
import helpers
movie_enc= helpers.load_pkl("../objs/enc_movie_with_meta+video.obj")
user_enc= helpers.load_pkl("../objs/enc_user.obj")

movie_shape= len(movie_enc.get(list(movie_enc.keys())[0]))
user_shape= len(user_enc.get(list(user_enc.keys())[0]))
```
