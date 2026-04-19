import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ImportProgress } from "../ImportProgress";

describe("ImportProgress", () => {
  it("shows Stage A reading state with memory counter", () => {
    render(
      <ImportProgress
        stage="stage_a"
        memoriesProcessed={1453}
        memoriesTotal={2143}
        entityCounts={{ people: 12, projects: 34, concepts: 89, decisions: 312, tools: 47 }}
        conceptsWritten={0}
        conceptsTotal={0}
        conceptTitles={[]}
      />,
    );
    expect(screen.getByText(/reading/i)).toBeInTheDocument();
    expect(screen.getByText(/1,453 \/ 2,143/)).toBeInTheDocument();
    expect(screen.getByText(/12 people/)).toBeInTheDocument();
    expect(screen.getByText(/34 projects/)).toBeInTheDocument();
  });

  it("shows Stage B compiling state with concept counter", () => {
    render(
      <ImportProgress
        stage="stage_b"
        memoriesProcessed={2143}
        memoriesTotal={2143}
        entityCounts={{ people: 47, projects: 56, concepts: 156, decisions: 412, tools: 89 }}
        conceptsWritten={3}
        conceptsTotal={47}
        conceptTitles={["Rust errors", "React hooks"]}
      />,
    );
    expect(screen.getByText(/compiling/i)).toBeInTheDocument();
    expect(screen.getByText(/3 \/ 47/)).toBeInTheDocument();
    expect(screen.getByText(/Rust errors/)).toBeInTheDocument();
  });

  it("shows done state with full summary", () => {
    render(
      <ImportProgress
        stage="done"
        memoriesProcessed={2143}
        memoriesTotal={2143}
        entityCounts={{ people: 47, projects: 56, concepts: 156, decisions: 412, tools: 89 }}
        conceptsWritten={47}
        conceptsTotal={47}
        conceptTitles={[]}
      />,
    );
    expect(screen.getByText(/concepts are ready/i)).toBeInTheDocument();
  });
});
