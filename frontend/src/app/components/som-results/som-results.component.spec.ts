import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SomResultsComponent } from './som-results.component';

describe('SomResultsComponent', () => {
  let component: SomResultsComponent;
  let fixture: ComponentFixture<SomResultsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SomResultsComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SomResultsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
