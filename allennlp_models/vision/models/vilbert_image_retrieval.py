# Vilbert takes in:
    # 1. Image token?
    # 2. image features
    # 3. clasification token
    # 4. word tokens
    # 5. separation token
    # (in multi-task training) 6. task token

# Vilbert outputs:
    # h_img, h_cls, h_sep
    # we will treat h_img and h_cls as "holistic image and text representations" (respectively)
    # to get a score, we:
        # 1. element-wise multiply h_img and h_cls
        # 2. then multiply the result by weights to get a number
        # 3. then softmax to get a probability?

# idea for instances:
    # offline, calculate 3 hard negatives for every caption/image pair
        # I think the hard negatives are *images*, not other captions
        # "For efficiency, you can calculate the L2 distance between image feature and caption feature and use that"
    # this will look like a multiple choice setup, I think