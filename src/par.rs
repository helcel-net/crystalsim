//! Parallel iteration over cells: rayon when the `native` feature is on,
//! plain sequential iterators otherwise (WebAssembly in a browser has no
//! threads without cross-origin isolation, which static hosts do not give).
//! The sequential shim mirrors the handful of rayon methods the crate uses,
//! so the growth code is written once.

#[cfg(feature = "native")]
pub use rayon::prelude::*;

#[cfg(not(feature = "native"))]
pub use seq::*;

#[cfg(not(feature = "native"))]
mod seq {
    use std::iter::Sum;

    /// A "parallel" iterator that runs in place.
    pub struct Seq<I>(pub I);

    impl<I: Iterator> IntoIterator for Seq<I> {
        type Item = I::Item;
        type IntoIter = I;
        fn into_iter(self) -> I {
            self.0
        }
    }

    impl<I: Iterator> Seq<I> {
        pub fn enumerate(self) -> Seq<std::iter::Enumerate<I>> {
            Seq(self.0.enumerate())
        }
        pub fn zip<J: IntoIterator>(self, other: J) -> Seq<std::iter::Zip<I, J::IntoIter>> {
            Seq(self.0.zip(other))
        }
        pub fn map<B, F: FnMut(I::Item) -> B>(self, f: F) -> Seq<std::iter::Map<I, F>> {
            Seq(self.0.map(f))
        }
        pub fn filter<P: FnMut(&I::Item) -> bool>(self, p: P) -> Seq<std::iter::Filter<I, P>> {
            Seq(self.0.filter(p))
        }
        pub fn for_each<F: FnMut(I::Item)>(self, f: F) {
            self.0.for_each(f)
        }
        pub fn collect<C: FromIterator<I::Item>>(self) -> C {
            self.0.collect()
        }
        pub fn sum<S: Sum<I::Item>>(self) -> S {
            self.0.sum()
        }
        pub fn reduce<ID: Fn() -> I::Item, OP: Fn(I::Item, I::Item) -> I::Item>(
            self,
            identity: ID,
            op: OP,
        ) -> I::Item {
            self.0.fold(identity(), op)
        }
        pub fn collect_into_vec(self, out: &mut Vec<I::Item>) {
            out.clear();
            out.extend(self.0);
        }
    }

    pub trait ParallelSliceMut<T> {
        fn par_chunks_mut(&mut self, size: usize) -> Seq<std::slice::ChunksMut<'_, T>>;
        fn par_iter_mut(&mut self) -> Seq<std::slice::IterMut<'_, T>>;
    }

    impl<T> ParallelSliceMut<T> for [T] {
        fn par_chunks_mut(&mut self, size: usize) -> Seq<std::slice::ChunksMut<'_, T>> {
            Seq(self.chunks_mut(size))
        }
        fn par_iter_mut(&mut self) -> Seq<std::slice::IterMut<'_, T>> {
            Seq(self.iter_mut())
        }
    }

    pub trait ParallelSlice<T> {
        fn par_iter(&self) -> Seq<std::slice::Iter<'_, T>>;
    }

    impl<T> ParallelSlice<T> for [T] {
        fn par_iter(&self) -> Seq<std::slice::Iter<'_, T>> {
            Seq(self.iter())
        }
    }

    pub trait IntoParallelIterator: IntoIterator + Sized {
        fn into_par_iter(self) -> Seq<Self::IntoIter> {
            Seq(self.into_iter())
        }
    }

    impl<I: IntoIterator> IntoParallelIterator for I {}
}
