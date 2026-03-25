box(ID,S,X,Y,W,H,I,T) :- pred(ID,S,X,Y,W,H,I,T).
invalid(ID,"Edge") :- box(ID,_,X,Y,W,H,_,_), X < 5.
invalid(ID,"Edge") :- box(ID,_,X,Y,W,H,_,_), Y < 5.
invalid(ID,"Edge") :- box(ID,_,X,Y,W,H,_,_), X+W > 507.
invalid(ID,"Flat") :- box(ID,_,_,_,W,H,_,_), W > 4*H.
invalid(ID,"Thin") :- box(ID,_,_,_,W,H,_,_), H > 4*W.
invalid(ID,"PureBlack") :- box(ID,_,_,_,_,_,I,_), I < 25.
invalid(ID,"PureWhite") :- box(ID,_,_,_,_,_,I,_), I > 95.
invalid(ID,"HighContrast") :- box(ID,_,_,_,_,_,_,T), T > 55.
zone_heart(ID) :- box(ID,_,X,Y,_,_,_,_), X > 240, X < 420, Y > 200.
zone_upper(ID) :- box(ID,_,_,Y,_,_,_,_), Y < 200.
zone_lower_safe(ID) :- box(ID,_,_,Y,_,_,_,_), Y >= 200, not zone_heart(ID).
safe_zone(ID) :- box(ID,_,X,Y,W,H,_,_), X > 30, Y > 30, X+W < 482, Y+H < 482.
intensity_ok(ID) :- box(ID,_,_,_,_,_,I,_), I > 35, I < 90.
is_cloudy(ID) :- box(ID,_,_,_,_,_,_,T), T > 10, T < 45.
valid(ID) :- box(ID,S,_,_,_,_,_,_), S >= 50, not invalid(ID,_).
valid(ID) :- box(ID,S,_,_,_,_,_,_), zone_lower_safe(ID), S >= 28, S < 50, not invalid(ID,_), safe_zone(ID), intensity_ok(ID).
valid(ID) :- box(ID,S,_,_,_,_,_,_), zone_heart(ID), S >= 34, S < 50, not invalid(ID,_), safe_zone(ID), intensity_ok(ID), is_cloudy(ID).
valid(ID) :- box(ID,S,_,_,_,_,_,_), zone_upper(ID), S >= 44, S < 50, not invalid(ID,_), safe_zone(ID), intensity_ok(ID).
#show valid/1.
