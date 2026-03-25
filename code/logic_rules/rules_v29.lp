box(ID,S,X,Y,W,H,I) :- pred(ID,S,X,Y,W,H,I,_).

invalid(ID,"Edge") :- box(ID,_,X,Y,W,H,_), X < 5.
invalid(ID,"Edge") :- box(ID,_,X,Y,W,H,_), Y < 5.
invalid(ID,"Edge") :- box(ID,_,X,Y,W,H,_), X+W > 507.
invalid(ID,"PureBlack") :- box(ID,_,_,_,_,_,I), I < 20.
invalid(ID,"PureWhite") :- box(ID,_,_,_,_,_,I), I > 95.

is_organic(ID) :- box(ID,_,_,_,W,H,_), 10*W < 16*H, 10*H < 16*W.
is_safe_side(ID) :- box(ID,_,X,_,_,_,_), X < 230.
is_lower(ID) :- box(ID,_,_,Y,_,_,_), Y >= 200.
safe_zone(ID) :- box(ID,_,X,Y,W,H,_), X > 30, Y > 30, X+W < 482, Y+H < 482.
intensity_ok(ID) :- box(ID,_,_,_,_,_,I), I > 35, I < 90.
size_ok(ID) :- box(ID,_,_,_,W,H,_), W*H > 2500.

valid(ID) :- box(ID,S,_,_,_,_,_), S >= 50, not invalid(ID,_).
valid(ID) :- box(ID,S,_,_,_,_,_), is_organic(ID), S >= 32, S < 50, not invalid(ID,_), safe_zone(ID), intensity_ok(ID), size_ok(ID).
valid(ID) :- box(ID,S,_,_,_,_,_), not is_organic(ID), is_safe_side(ID), is_lower(ID), S >= 35, S < 50, not invalid(ID,_), safe_zone(ID), intensity_ok(ID).

#show valid/1.