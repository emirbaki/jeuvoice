//! The CPU part of Whisper decoding: the vocab, the token rules, the window loop, the token
//! times and DTW. It is a port of whisper.cpp and needs no GPU.

pub mod vocab;
