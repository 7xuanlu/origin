// Hand-written typings for the vendored flextree.js (only the surface we use).

export interface FlexNode<Datum> {
  data: Datum;
  depth: number;
  /** breadth-axis center, in whatever unit nodeSize returns */
  x: number;
  /** depth-axis top edge */
  y: number;
  xSize: number;
  ySize: number;
  children: FlexNode<Datum>[] | null;
  descendants(): FlexNode<Datum>[];
  each(cb: (node: FlexNode<Datum>) => void): this;
}

export interface FlextreeOptions<Datum> {
  children?: (d: Datum) => Datum[] | undefined;
  /** [breadth, depth] size; receives the hierarchy node (n.data is your datum) */
  nodeSize?: (node: FlexNode<Datum>) => [number, number];
  spacing?: number | ((a: FlexNode<Datum>, b: FlexNode<Datum>) => number);
}

export interface FlextreeLayout<Datum> {
  /** mutates the hierarchy in place, setting x/y on every node */
  (tree: FlexNode<Datum>): FlexNode<Datum>;
  hierarchy(
    data: Datum,
    children?: (d: Datum) => Datum[] | undefined,
  ): FlexNode<Datum>;
}

export default function flextree<Datum>(
  options?: FlextreeOptions<Datum>,
): FlextreeLayout<Datum>;
